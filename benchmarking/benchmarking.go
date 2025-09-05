/*
 * Copyright (c) 2018 XLAB d.o.o
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright (c) 2018 XLAB d.o.o
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/big"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/JasZal/gofe/data"
	"github.com/JasZal/gofe/innerprod/noisy"
)

func UNUSED(...interface{}) {}

// performs benchmarking on dyno and stores it in a file
// if uncommented also measures times for diffpipe
// rounds: describes the number of rounds that time is averaged about
// distr: describes how the data set is distritbuted between n and m, possible values: 0, 0.5 and 1
func main() {
	results := "resultsReviews.txt"
	prefix := "./datasets/"
	postfix := ".txt"
	format := "millisec"
	conversion := make(map[string]float64)
	conversion["sec"] = 1000000000.0
	conversion["millisec"] = 1000000.0
	conversion["nanosec"] = 1.0

	//comparison NMCFE_OT with DiffPIPE
	datapoints := []int{100, 10000, 1000000}
	boundX := int(math.Pow(2, 16))
	boundY := int(math.Pow(2, 7))

	distribution := []float64{0, 0.5} //, 1}
	rounds := 50
	nrWorkers := runtime.NumCPU()

	//generateData(datapoints, boundX, boundY)
	write(results, "benchmarking Dyno, DiffPIPE\nTimes in "+format+"\nDS n,m,Setup, Enc, KeyGen, Dec\n", false)
	write(results, "NMMCFE\n", true)

	for _, dp := range datapoints {
		for _, distr := range distribution {
			//set n and m
			var n, m int
			if distr == 0 {
				n = 1
				m = dp
			} else if distr == 0.5 {
				n = int(math.Sqrt(float64(dp)))
				m = int(math.Sqrt(float64(dp)))

			} else if distr == 1 {
				n = dp
				m = 1
			} else {
				fmt.Println("distribution not supported")
			}

			//read data and func
			plaintexts := loadData(prefix+"data_"+fmt.Sprint(dp)+postfix, n, m)
			yplain := loadData(prefix+"func_"+fmt.Sprint(dp)+postfix, n, m)

			//test dyno
			tS, tE, tKG, tD := 0.0, 0.0, 0.0, 0.0
			for r := 0; r < rounds; r++ {
				timeS, timeE, timeKG, timeD := NMCFE(plaintexts, yplain, boundX, boundY, nrWorkers)

				tS += float64(timeS.Nanoseconds()) / float64(rounds)
				tE += float64(timeE.Nanoseconds()) / float64(rounds)
				tKG += float64(timeKG.Nanoseconds()) / float64(rounds)
				tD += float64(timeD.Nanoseconds()) / float64(rounds)
			}

			fmt.Printf("Dyno: setup: %.3f %v, enc %.3f %v, keygen %.3f %v, dec %.3f %v\n", tS/conversion[format], format, tE/conversion[format], format, tKG/conversion[format], format, tD/conversion[format], format)

			//write results
			write(results, fmt.Sprintf(" %v & %v& %.3f&%.3f&%.3f&%.3f\\\\\n", n, m, tS/conversion[format], tE/conversion[format], tKG/conversion[format], tD/conversion[format]), true)

			//test DiffPIPE
			//please uncomment if this times should be included

			// tS, tE, tKG, tD = 0.0, 0.0, 0.0, 0.0
			// for r := 0; r < rounds; r++ {
			// 	timeS, timeE, timeKG, timeD := DiffPIPE(plaintexts, yplain, boundX, boundY, nrWorkers)
			// 	tS += float64(timeS.Nanoseconds()) / float64(rounds)
			// 	tE += float64(timeE.Nanoseconds()) / float64(rounds)
			// 	tKG += float64(timeKG.Nanoseconds()) / float64(rounds)
			// 	tD += float64(timeD.Nanoseconds()) / float64(rounds)
			// }

			// fmt.Printf("DiffPIPE: setup: %.3f %v, enc %.3f %v, keygen %.3f %v, dec %.3f %v\n", tS/conversion[format], format, tE/conversion[format], format, tKG/conversion[format], format, tD/conversion[format], format)

			// //write results
			// write(results, fmt.Sprintf("DiffPIPE: %v, %v, %.3f,%.3f,%.3f,%.3f\n", n, m, tS/conversion[format], tE/conversion[format], tKG/conversion[format], tD/conversion[format]), true)

		}

	}

	UNUSED(datapoints, results)
}

// computes runtimes for Diffpipe on given data set with given function
func DiffPIPE(plaintexts, yplain []data.Vector, boundX, boundY, nrWorkers int) (time.Duration, time.Duration, time.Duration, time.Duration) {
	var tSetup, tEnc, tKeyGen, tDec time.Duration
	secLevel := 2
	clients := len(plaintexts)
	attributes := len(plaintexts[0])
	fmt.Println("Diffpipe")

	//generate Authority with Params and Keys
	start := time.Now()
	fe := noisy.NewOTNHMultiIPE(secLevel, clients, attributes, big.NewInt(int64(boundX)), big.NewInt(int64(boundY)))
	msk, pk, _ := fe.GenerateKeys()
	tSetup = time.Since(start)
	fmt.Println("finished Setup")
	//	encrypt vectors
	wg := sync.WaitGroup{}
	cipher := make(data.MatrixG1, clients)

	input := make(chan int)
	start = time.Now()

	for i := 0; i < nrWorkers; i++ {
		wg.Add(1)
		go workersDiffPIPE(plaintexts, cipher, msk, fe, input, &wg)

	}
	for i := 0; i < clients; i++ {
		input <- i
	}

	close(input)

	wg.Wait()
	tEnc = time.Since(start)
	fmt.Println("finished encryption")
	// derive a functional key for matrix y
	start = time.Now()
	key, err := fe.DeriveKey(yplain, msk, 0)
	tKeyGen = time.Since(start)
	if err != nil {
		fmt.Printf("Error during derive key: %v", err)
	}

	// decrypt
	start = time.Now()
	_, err = fe.Decrypt(cipher, key, pk)
	//fmt.Printf("res DiffPIPE: %v\n", res)
	tDec = time.Since(start)

	if err != nil {

		log.Fatalf("Error during decryption: %v", err)
	}

	return tSetup, tEnc, tKeyGen, tDec
}

// computes runtimes for Dyno on given data set with given function
func NMCFE(plaintexts, yplain []data.Vector, boundX, boundY, nrWorkers int) (time.Duration, time.Duration, time.Duration, time.Duration) {
	var tSetup, tEnc, tKeyGen, tDec time.Duration

	// build the scheme
	start := time.Now()
	numClient := len(yplain)
	vecLen := len(yplain[0])
	fe := noisy.NewOTPRF(numClient, vecLen, big.NewInt(int64(boundX)), big.NewInt(int64(boundX)), big.NewInt(0))

	// generate master secret key
	masterSecKey := fe.GenerateKeys()
	tSetup = time.Since(start)

	//encrypt vectors
	wg := sync.WaitGroup{}
	label := make([]byte, 16)
	cipher := make([]data.Vector, numClient)

	input := make(chan int)
	start = time.Now()

	for i := 0; i < nrWorkers; i++ {
		wg.Add(1)
		go workersMCFE(plaintexts, label, cipher, masterSecKey, fe, input, &wg)

	}
	for i := 0; i < numClient; i++ {
		input <- i
	}

	close(input)

	wg.Wait()
	tEnc = time.Since(start)

	// derive a functional key for matrix y
	start = time.Now()
	key, err := fe.DeriveKey(masterSecKey, yplain, big.NewInt(0), label, nrWorkers)
	tKeyGen = time.Since(start)
	if err != nil {
		fmt.Printf("Error during derive key: %v\n", err)
	}

	// decrypt
	start = time.Now()
	_, err = fe.Decrypt(cipher, key, yplain)

	if err != nil {
		fmt.Printf("Error during decryption:  %v\n", err)
	}
	//	fmt.Printf("res MCFE: %v\n", res)
	tDec = time.Since(start)

	if err != nil {

		log.Fatalf("Error during decryption: %v", err)
	}

	return tSetup, tEnc, tKeyGen, tDec

}

func workersMCFE(x []data.Vector, label []byte, cipher []data.Vector, masterSecKey [][]byte, fe *noisy.OTPRF, input chan int, wg *sync.WaitGroup) {

	defer wg.Done()
	var err error

	for i := range input {
		cipher[i], err = fe.Encrypt(x[i], label, masterSecKey[i])

		if err != nil {
			log.Fatalf("Error during encryption: %v", err)
		}
	}
}

func workersDiffPIPE(x []data.Vector, cipher data.MatrixG1, msk *noisy.OTNHMultiIPESecKey, fe *noisy.OTNHMultiIPE, input chan int, wg *sync.WaitGroup) {

	defer wg.Done()
	var err error

	for i := range input {
		cipher[i], err = fe.Encrypt(x[i], msk.BHat[i])
		if err != nil {
			log.Fatalf("Error during encryption: %v", err)
		}
	}
}

// writes results into file
func write(filename string, message string, append bool) {

	var file *os.File
	var err error

	if append {
		file, err = os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0755)
	} else {
		file, err = os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0755)
	}

	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	_, err = file.WriteString(message)
	if err != nil {
		log.Fatal(err)
	}

}

// generates data for benchmarking
func generateData(datapoints []int, boundX, boundY int) {
	//data sets have 10000 attr per line

	prefix := "./datasets/"
	postfix := ".txt"

	for _, dp := range datapoints {
		data := make([]string, dp)
		y := make([]string, dp)

		for i := 0; i < dp; i++ {
			data[i] = fmt.Sprint(rand.Intn(int(boundX)))
			y[i] = fmt.Sprint(rand.Intn(boundY))
		}
		if dp < 10000 {
			write(prefix+"data_"+fmt.Sprint(dp)+postfix, strings.Join(data, ", "), false)
			write(prefix+"func_"+fmt.Sprint(dp)+postfix, strings.Join(y, ", "), false)

		} else {
			write(prefix+"data_"+fmt.Sprint(dp)+postfix, "", false)
			write(prefix+"func_"+fmt.Sprint(dp)+postfix, "", false)
			for i := 0; i < dp/10000; i++ {
				write(prefix+"data_"+fmt.Sprint(dp)+postfix, strings.Join(data[10000*i:10000*(i+1)], ", ")+"\n", true)
				write(prefix+"func_"+fmt.Sprint(dp)+postfix, strings.Join(y[10000*i:10000*(i+1)], ", ")+"\n", true)
			}
		}
	}
}

// loads data for benchmarking
func loadData(filename string, n, m int) []data.Vector {
	// Open the file
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Read the original matrix
	var values []int64
	scanner := bufio.NewScanner(file)
	const maxBufferSize = 1024 * 1024
	scanner.Buffer(make([]byte, maxBufferSize), maxBufferSize)
	nOld := 0
	mOld := 0
	for scanner.Scan() {
		nOld++
		line := scanner.Text()
		nums := strings.Fields(line)
		mOld = len(nums)
		for _, num := range nums { // Remove any commas or whitespace
			num = strings.TrimSpace(strings.ReplaceAll(num, ",", ""))
			val, err := strconv.Atoi(num)
			if err != nil {
				log.Fatal(err)
			}
			values = append(values, int64(val))
		}

	}

	// Check if an error occurred during scanning
	if err := scanner.Err(); err != nil {
		fmt.Println("Error while scanning:", err)
	}

	// Check if the number of values matches the new matrix dimensions
	if len(values) != n*m {
		log.Fatal(fmt.Sprintf("new dimension does not fit to data set (old n=%v, m=%v,  new n=%v, m=%v)", nOld, mOld, n, m))
	}

	// Reshape into a new matrix
	matrix := make([]data.Vector, n)

	for i := 0; i < n; i++ {
		matrix[i] = data.NewConstantVector(m, big.NewInt(0))
		for j := 0; j < m; j++ {
			matrix[i][j] = big.NewInt(values[i*m+j])
		}
	}
	return matrix
}

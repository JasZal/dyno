package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/big"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/JasZal/gofe/data"
	"github.com/JasZal/gofe/innerprod/noisy"
)

var deb bool = true

func debug(s string) {
	if deb {
		fmt.Println(s)
	}
}

// UNUSED allows unused variables to be included in Go programs
func UNUSED(x ...interface{}) {}

// This method trains a logistic regrestion and stores results in the specified file
// Following variables can be set:
// scaling: 			scaling factor
// it: 	number of training iterations
// epsilon: 		for DP (delta is set to 1/#rec) (does not have an affect on the runtime)
// rounds: how many rounds this should run
func main() {

	fmt.Println("started training and benchmarking on data sets")

	scaling := 1000000000
	boundX := big.NewInt(int64(1 * scaling))
	boundY := big.NewInt(int64(10 * scaling))
	boundN := big.NewInt(int64(1 * float64(scaling)))

	noisyB := true

	// attributes
	rounds := 1
	it := []int{50}
	nrWorkers := runtime.NumCPU()

	var epsilon []float64
	for i := 0.00; i <= 5.0; i += 0.5 {
		if i == 0.0 {
			epsilon = append(epsilon, 0.01)
		} else {
			epsilon = append(epsilon, i)
		}
	}
	epsilon = []float64{5.0}

	filePrefix := "./datasets/training"
	files := []string{"LBW.csv", "PCS.csv", "UIS.csv", "Nhanes.csv"}

	//batchsize and alpha in dependency of data set
	alphaF := [][]float64{[]float64{0.1}, []float64{0.1, 0.3, 0.6, 0.9}, []float64{0.1}}

	for _, iterations := range it {

		fileRes := "results" + fmt.Sprint(iterations) + ".txt"

		write(fileRes, "time in Nanosec\n", true)
		write(fileRes, "eps = ", true)
		write(fileRes, fmt.Sprintln(epsilon), true)

		for fI, file := range files {
			alphaS := alphaF[fI]

			debug(fmt.Sprintf("****************file: %v**********************\n", file))
			write(fileRes, fmt.Sprintf(file), true)
			write(fileRes, "= [ ", true)

			// read data
			file = filePrefix + file
			attr, m, dataPlain, keys := loadData(file, scaling)
			testData := loadTestData(file)
			debug("data loaded")

			n := len(dataPlain)

			batchsize := n
			delta := 1.0 / float64(n)

			timeTotal := 0.0

			for _, eps := range epsilon {
				max := make([]float64, 2)
				fmt.Printf("e= %v\n", eps)
				for _, alpha := range alphaS {

					acc := 0.0
					for r := 0; r < rounds; r++ {

						//setup scheme/authority
						a, tSetupA := NewAuthority(m, n, boundX, boundY, boundN, eps, delta, int64(scaling), keys, nrWorkers, noisyB)
						//debug("time Setup: " + tSetupA.String())

						//setup clients/encrypt data
						ct := make([]data.Vector, n)
						wg := sync.WaitGroup{}

						chIn := make(chan int)
						startE := time.Now()

						for i := 0; i < nrWorkers; i++ {
							wg.Add(1)
							go func(chIn chan int) {
								defer wg.Done()
								var err error
								for i := range chIn {
									client := noisy.NewOTPRFFromParams(a.getParams())

									label := make([]byte, 16)
									start := time.Now()

									ct[i], err = client.Encrypt(dataPlain[i], label, a.getEncryptionKey(i))
									timeEnc := time.Since(start)
									if err != nil {
										log.Fatalf("Error during encryption: %v", err)
									}
									if i == 0 {
										//debug("time Enc one rec: " + timeEnc.String())
										UNUSED(timeEnc, tSetupA)
									}
								}
							}(chIn)

						}

						for i := 0; i < n; i++ {
							chIn <- i
						}

						close(chIn)
						wg.Wait()

						tE := time.Since(startE)
						UNUSED(tE)
						fmt.Printf("time Encryption total: %v\n", tE)

						// setup evaluator
						e := NewEvaluator(int(attr), n, scaling, ct, a, batchsize, eps, delta)
						// start training
						theta, tReg, err := e.trainLogReg(iterations, alpha)
						fmt.Printf("theta: %v\n", theta)

						if err != nil {
							fmt.Printf("Runtime: %v\n", tReg)
							log.Fatal("Error during Training:", err)
						}

						acc += compAcc(testData, theta) / float64(rounds)
						timeTotal += float64(tReg.Nanoseconds()) / float64(rounds*len(epsilon)*len(alphaS))

						//debug(fmt.Sprintf("main: theta: %v\n", theta))
						//debug(fmt.Sprintf("main:Time LogReg: %v\n", tLinReg))
					}

					if acc >= max[1] {

						max[0] = alpha
						max[1] = acc
					}
					debug(fmt.Sprintf("alpha %v, accuracy: %v\n", alpha, acc))

				}

				write(fileRes, fmt.Sprintf("%v, ", max[1]), true)
				fmt.Printf("-- max: %v\n", max)
			}
			debug(fmt.Sprintf("average time LogReg: %v\n", timeTotal))
			write(fileRes, fmt.Sprintf("]\n"), true)
			write(fileRes, fmt.Sprintf(" av time: %v\n", timeTotal), true)

		}

	}

}

// writes result into file
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

// load test data from csv files
func loadTestData(file string) [][]float64 {
	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	dataR := csv.NewReader(f)
	records, err := dataR.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	dataT := make([][]float64, len(records))
	for i := 0; i < len(records); i++ {
		dataT[i] = make([]float64, len(records[0]))

		for j := 0; j < len(records[0]); j++ {
			interm, _ := strconv.ParseFloat(records[i][j], 64)

			dataT[i][j] = interm

		}
	}
	return dataT
}

// load training data from csv files
func loadData(file string, scaling int) (float64, int, data.Matrix, map[string]int) {

	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	dataR := csv.NewReader(f)
	records, err := dataR.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	attr := int64(len(records[0]) - 1)

	//blow up because of precomputed monomials (1/24 * m⁴ + 5/12 * m^3 + 35/24*m^2 + 37/12* m + 2)
	m := int(math.Round(math.Pow(float64(attr), 4.0)/24 + math.Pow(float64(attr), 3.0)*5/12 + math.Pow(float64(attr), 2.0)*35/24 + float64(attr)*37/12 + 2.0))

	x := make([][]float64, len(records))
	for i := 0; i < len(x); i++ {
		x[i] = make([]float64, len(records[0]))
		for j := 0; j < len(x[0]); j++ {
			interm, _ := strconv.ParseFloat(records[i][j], 64)
			x[i][j] = interm
		}
	}

	//create database as we want it, with kroenecker product
	dataX := make(data.Matrix, len(records))
	indices := make(map[string]int)

	b := big.NewInt(8)

	var i, j, k, v int64

	for r := 0; r < len(records); r++ {
		dataX[r] = make(data.Vector, m)
		lookup := map[string]float64{
			"0": 1,
		}

		for i = 0; i < attr; i++ {

			lookup[new(big.Int).Exp(b, big.NewInt(i), nil).String()] = x[r][i]

			for j = 0; j < attr; j++ {
				lookup[new(big.Int).Add(new(big.Int).Exp(b, big.NewInt(i), nil), new(big.Int).Exp(b, big.NewInt(j), nil)).String()] = x[r][i] * x[r][j]

				for k = 0; k < attr; k++ {

					lookup[new(big.Int).Add(new(big.Int).Add(new(big.Int).Exp(b, big.NewInt(i), nil), new(big.Int).Exp(b, big.NewInt(j), nil)), new(big.Int).Exp(b, big.NewInt(k), nil)).String()] = x[r][i] * x[r][j] * x[r][k]

					for v = 0; v < attr; v++ {
						lookup[new(big.Int).Add(new(big.Int).Add(new(big.Int).Exp(b, big.NewInt(i), nil), new(big.Int).Exp(b, big.NewInt(j), nil)), new(big.Int).Add(new(big.Int).Exp(b, big.NewInt(k), nil), new(big.Int).Exp(b, big.NewInt(v), nil))).String()] = x[r][i] * x[r][j] * x[r][k] * x[r][v]

					}
				}
			}
		}
		i, j, k, v = 0, 0, 0, 0
		//label
		for i = 0; i < attr; i++ {

			lookup[new(big.Int).Add(new(big.Int).Exp(b, big.NewInt(attr), nil), big.NewInt(i)).String()] = x[r][i] * x[r][attr]
		}

		lookup[new(big.Int).Add(new(big.Int).Exp(b, big.NewInt(attr), nil), big.NewInt(attr+1)).String()] = x[r][attr]

		keys := make([]string, 0)
		for k, _ := range lookup {
			keys = append(keys, k)
		}

		sort.Strings(keys)

		for i, k := range keys {
			dataX[r][i] = big.NewInt(int64(math.Round(lookup[k] * float64(scaling))))
			indices[k] = i
		}

	}

	return float64(len(records[0]) - 1), m, dataX, indices
}

// compute accuracy on testdata for given model weights theta
func compAcc(data [][]float64, theta []float64) float64 {

	sum := 0.0
	for i := 0; i < len(data); i++ {
		x := 0.0
		if h(data[i][0:len(data[0])-1], theta) >= 0.5 {
			x = 1.0
		}

		if x == data[i][len(data[0])-1] {
			sum += 1
		}

	}

	return sum / float64(len(data))

}

func h(x, theta []float64) float64 {

	sum := theta[len(x)]
	for i := 0; i < len(x); i++ {
		sum += x[i] * (theta[i])
	}

	return 0.5 + 1.20096*sum/8 - 0.81562*math.Pow(sum, 3)/512
}

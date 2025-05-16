package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/google/differential-privacy/go/noise"
)

func UNUSED(x ...interface{}) {}

var noisy bool
var approx bool
var degree int
var scaling float64

func main() {

	scaling = 0.0
	noisy = true
	approx = true
	if approx {
		scaling = 1000000.0
	}

	degree = 3

	rounds := 5

	var epsilon []float64
	for i := 0.00; i <= 8.0; i += 0.1 {
		if i == 0.0 {
			epsilon = append(epsilon, 0.01)
		} else {
			epsilon = append(epsilon, i)
		}
	}
	//epsilon = []float64{1}

	filePrefix := "../datasets/training"
	files := []string{"LBW", "PCS", "UIS"}
	it := []int{50}
	//files := []string{"Nhanes"} //
	//it := []int{50, 100, 150}
	filePostfix := ".csv"

	alphaF := map[string][]float64{
		"LBW":    []float64{0.01, 0.03, 0.06, 0.1, 0.3},
		"PCS":    []float64{0.01, 0.03, 0.06, 0.1, 0.3, 0.6},
		"UIS":    []float64{0.06, 0.1, 0.3, 0.6, 0.9},
		"Nhanes": []float64{0.1, 0.3, 0.6, 0.9},
	}
	fileRes := "results" + fmt.Sprint(scaling) + ".txt"
	batchsizesN := []int{0}
	write(fileRes, "eps = ", true)
	write(fileRes, fmt.Sprintln(epsilon), true)

	for _, file := range files {
		if file == "Nhanes" {
			fileRes = "resultsNhanes" + fmt.Sprint(scaling) + ".txt"
		}
		alphaS := alphaF[file]
		// read data
		file = filePrefix + file + filePostfix
		data := loadData(file)

		n := len(data)

		attr := len(data[0]) - 1

		//write(fileRes, "time in Nanosec\n", true)
		for _, iterations := range it {
			batch := 0
			for _, batch = range batchsizesN {
				if batch == 0 {
					batch = n
				}

				fmt.Printf("****************file: %v, batch: %v, iterations: %v **********************\n", file, batch, iterations)
				write(fileRes, fmt.Sprintf(file+fmt.Sprint(iterations)+"_"+fmt.Sprint(batch)+"= [ "), true)

				del := 1.0 / float64(n)
				for _, eps := range epsilon {
					max := make([]float64, 2)
					fmt.Printf("e= %v\n", eps)
					for _, alpha := range alphaS {
						acc := 0.0
						for r := 0; r < rounds; r++ {
							theta0 := make([]float64, attr)
							theta, err := gradientDescent(data, iterations, batch, alpha, eps, del, theta0)
							if err != nil {
								//fmt.Println("log reg did not converge with alpha: ", alpha)
								break
							}
							acc += compAcc(data, theta) / float64(rounds)
						}
						if acc > max[1] {

							max[0] = alpha
							max[1] = acc
						}

						//nur Nhanes
						if file == "Nhanes" {
							if max[0] == 0.9 {
								alphaS = []float64{0.6, 0.9}
							}
						}

					}
					write(fileRes, fmt.Sprintf("%v, ", max[1]), true)
					fmt.Printf("-- max: %v\n", max)
				}
				write(fileRes, fmt.Sprintf("];\n"), true)
			}
		}
	}

}

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

func loadData(file string) [][]float64 {
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

	data := make([][]float64, len(records))
	for i := 0; i < len(records); i++ {
		data[i] = make([]float64, len(records[0])+1)
		data[i][0] = 1

		for j := 0; j < len(records[0]); j++ {
			interm, _ := strconv.ParseFloat(records[i][j], 64)

			data[i][j+1] = interm
		}
	}
	return data
}

func sigmoid(x float64) float64 {
	if !approx {
		return 1 / (1 + math.Exp(-1*x))
	} else {
		//approx sigmoid:
		if degree == 3 {
			//degree 3		//0.5 - 1.20096*(-x/8) + 0.81562*(-x/8)^3
			return 0.5 + 1.20096*x/8 - 0.81562*math.Pow(x, 3)/512
		} else {
			if degree == 1 {
				// degree 1
				return 0.5 + 0.25*x
			} else {
				fmt.Println("no such approximation implemented")
				return math.NaN()
			}
		}

	}
}

func hGD(x, theta []float64, a_b float64) float64 {
	sum := 0.0
	for i := 0; i < len(x); i++ {
		if scaling != 0.0 {
			sum += math.Round(x[i]*scaling) / scaling * math.Round(theta[i]*scaling) / scaling
		} else {
			sum += x[i] * (theta[i])
		}

	}

	sum = sigmoid(sum)
	sum *= a_b
	if scaling != 0.0 {
		sum = math.Round(sum*math.Pow(scaling, 2)) / math.Pow(scaling, 2)
	}

	return sum

}

func h(x, theta []float64) float64 {

	sum := 0.0
	for i := 0; i < len(x); i++ {
		if scaling != 0.0 {
			sum += math.Round(x[i]*scaling) / scaling * (theta[i])
		} else {
			sum += x[i] * (theta[i])
		}

	}

	return sigmoid(sum)
}

func computeInfSen(theta []float64, alpha, n float64) float64 {

	sumT := 0.0
	for i := 0; i < len(theta); i++ {
		sumT += math.Abs(theta[i])
	}

	if degree == 1 {
		return alpha / n * (1 + 0.25*sumT)
	} else if degree == 3 {
		a3 := -0.81562 / 512
		a1 := -1.20086 / 8
		return alpha / n * (1 + math.Abs(a3*math.Pow(sumT, 3)-a1*sumT))
	} else {
		log.Fatal("no such approximation known")
		return math.NaN()
	}

}

func compute0Sen(theta []float64) int64 {
	return int64(len(theta))
}

func computeNoise(theta []float64, alpha, b, eps, del float64) ([]float64, error) {
	n := make([]float64, len(theta))

	infSen := computeInfSen(theta, alpha, b)
	if math.IsInf(infSen, 0) || math.IsNaN(infSen) {
		err := fmt.Errorf("infSen too big or Nan")
		//fmt.Printf("err: %v\n", err)
		return nil, err
	}
	zeroSen := compute0Sen(theta)

	//noise via gauss

	gauss := noise.Gaussian()
	for j := 0; j < len(n); j++ {
		n[j] = gauss.AddNoiseFloat64(0.0, zeroSen, infSen, eps, del)
	}
	for j := 0; j < len(n); j++ {
		n[j] *= math.Pow(scaling, 2)

		if n[j] >= 0 {
			n[j] = math.Ceil(n[j])
		} else {
			n[j] = math.Floor(n[j])
		}

		n[j] = n[j] / math.Pow(scaling, 2)
	}

	return n, nil
}

func gradientDescentIteration(data [][]float64, alpha, eps, delta float64, theta []float64) ([]float64, error) {
	var a_b float64
	noise := make([]float64, len(theta))
	thetaN := make([]float64, len(theta))

	a_b = alpha / float64(len(data))
	if scaling != 0.0 {
		a_b = math.Round(a_b*scaling) / scaling
	}

	if noisy {
		var err error
		noise, err = computeNoise(theta, alpha, float64(len(data)), eps, delta)
		if err != nil {
			return nil, err
		}
	}

	for j := 0; j < len(theta); j++ {
		sum := 0.0
		for i := 0; i < len(data); i++ {
			if scaling != 0.0 {
				sum += (a_b*math.Round(data[i][len(data[i])-1]*data[i][j]*scaling)/scaling - math.Round((hGD(data[i][0:len(data[i])-1], theta, a_b)*(math.Round(data[i][j]*scaling)/scaling))*math.Pow(scaling, 2))/math.Pow(scaling, 2))
			} else {
				sum += (a_b*data[i][len(data[i])-1] - hGD(data[i][0:len(data[i])-1], theta, a_b)) * data[i][j]
			}

		}

		thetaN[j] = theta[j] + sum + noise[j]

	}

	return thetaN, nil
}

func gradientDescent(data [][]float64, it, batchs int, alpha, eps, del float64, theta []float64) ([]float64, error) {
	var err error
	for i := 0; i < it; i++ {

		eI := eps*math.Pow(float64(i+1)/float64(it), 1.5) - eps*math.Pow(float64(i)/float64(it), 1.5)

		if batchs == len(data) {
			theta, err = gradientDescentIteration(data, alpha, eI, del/float64(it), theta)
			if err != nil {
				return nil, err
			}
		} else {
			bstart := (i * batchs) % len(data)
			bend := ((i + 1) * batchs) % len(data)
			var batch [][]float64

			if bstart <= bend {
				batch = data[bstart:bend]
			} else {
				batch = append(data[bstart:], data[:bend]...)
			}

			theta, err = gradientDescentIteration(batch, alpha, eI, del/float64(it), theta)
			if err != nil {
				return nil, err
			}

		}

		//fmt.Printf("it %v, theta: %v\n", i, theta)
	}

	return theta, nil

}

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

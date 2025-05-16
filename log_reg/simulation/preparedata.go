package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

func normalize(data [][]string) [][]string {

	attr := len(data[0])
	min := make([]float64, attr)
	max := make([]float64, attr)

	for j := 0; j < attr; j++ {
		for i := 0; i < len(data); i++ {
			interm, _ := strconv.ParseFloat(data[i][j], 64)
			if interm < min[j] {
				min[j] = interm
			}
			if interm > max[j] {
				max[j] = interm
			}
		}
	}

	res := make([][]string, len(data))

	for i := 0; i < len(data); i++ {

		res[i] = make([]string, len(data[0]))
		for j := 0; j < attr; j++ {
			interm, _ := strconv.ParseFloat(data[i][j], 64)
			interm = (interm - min[j]) / (max[j] - min[j])
			res[i][j] = fmt.Sprint(interm)
		}
	}

	return res

}

func prepareData(source, filetr string) {

	f, err := os.Open(source)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	dataR := csv.NewReader(f)
	records, err := dataR.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// shuffled := make([][]string, len(records))
	// perm := rand.Perm(len(records))
	// for i, v := range perm {
	// 	shuffled[v] = records[i]
	// }

	//assuming cliped file and normalize + scale inputs
	shuffled := normalize(records)

	// split the training set
	trainingIdx := (len(shuffled)) // * 4 / 5
	trainingSet := shuffled[0:trainingIdx]
	// split the testing set
	//testingSet := shuffled[trainingIdx+1:]
	// we write the splitted sets in separate files

	sets := map[string][][]string{
		filetr: trainingSet,
		//"../datasets/testing.csv":   testingSet,
	}

	for fn, dataset := range sets {
		f, err := os.Create(fn)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		out := csv.NewWriter(f)
		if err := out.WriteAll(dataset); err != nil {
			log.Fatal(err)
		}
		out.Flush()
	}
}

func computeLabelLin(data [][]string, theta []float64) [][]string {

	for i := 0; i < len(data); i++ {

		y := 0.0

		for j := 0; j < len(data[0])-1; j++ {
			interm, _ := strconv.ParseFloat(data[i][j], 64)
			y += interm * theta[j]
		}

		data[i][len(data[0])-1] = fmt.Sprint(y)

	}

	return data
}

func computeLabelLog(data [][]string, theta []float64) [][]string {

	for i := 0; i < len(data); i++ {

		sum := 0.0

		for j := 0; j < len(data[0])-1; j++ {
			interm, _ := strconv.ParseFloat(data[i][j], 64)
			sum += interm * theta[j]
		}

		y := 1.0 / (1.0 + math.Exp(-1.0*sum))

		if y < 0.5 {
			data[i][len(data[0])-1] = "0"
		} else {
			data[i][len(data[0])-1] = "1"
		}
	}

	return data
}

func fakeDataSet(theta []float64, n int, file string) {
	records := make([][]string, n)
	for i := 0; i < n; i++ {
		records[i] = make([]string, len(theta))
		for j := 0; j < len(theta); j++ {
			records[i][j] = fmt.Sprintf("%v", rand.Float64())
		}
	}

	// fake label according to theta
	computeLabelLog(records, theta)

	sets := map[string][][]string{
		file: records,
	}

	for fn, dataset := range sets {
		f, err := os.Create(fn)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		out := csv.NewWriter(f)
		if err := out.WriteAll(dataset); err != nil {
			log.Fatal(err)
		}
		out.Flush()
	}
}

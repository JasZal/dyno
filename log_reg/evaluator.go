package main

import (
	"fmt"
	"log"
	"math"
	"time"

	"github.com/JasZal/gofe/data"
	"github.com/JasZal/gofe/innerprod/noisy"
)

type Evaluator struct {
	m         int
	n         int
	scaling   int
	batchsize int
	fe        *noisy.OTPRF
	cts       []data.Vector
	epsilon   float64
	delta     float64
	a         *Authority
}

func NewEvaluator(attr, numC, scaling int, cts []data.Vector, a *Authority, b int, eps, d float64) *Evaluator {
	e := &Evaluator{
		m:         attr,
		n:         numC,
		fe:        a.fe,
		cts:       cts,
		a:         a,
		epsilon:   eps,
		delta:     d,
		scaling:   scaling,
		batchsize: b,
	}

	return e
}

func (e Evaluator) trainLogReg(iterations int, alpha float64) ([]float64, time.Duration, error) {
	start := time.Now()
	theta := make([]float64, e.m+1)

	// for i := 0; i < len(theta); i++ {
	// 	theta[i] = rand.Float64()
	// }
	//	fmt.Printf("theta: %v\n", theta)
	eI := 0.0
	for i := 0; i < iterations; i++ {

		eps := e.epsilon*math.Pow(float64(i+1)/float64(iterations), 1.5) - e.epsilon*math.Pow(float64(i)/float64(iterations), 1.5)
		eI += eps
		del := e.delta / float64(iterations)

		//compute batch
		bstart := (i * e.batchsize) % len(e.cts)

		batch := make([]int, e.batchsize)
		for j := 0; j < e.batchsize; j++ {
			batch[j] = (bstart + j) % len(e.cts)
		}

		startI := time.Now()

		tDK := time.Now()
		dk, y := e.a.generateDK(theta, e.m, batch, eps, del, alpha)
		//debug(fmt.Sprintf("time generating DK: %v \n ", time.Since(tDK)))

		for j := 0; j < e.m+1; j++ {
			f := noisy.NewOTPRFFromParams(e.fe.Params)

			res, err := f.Decrypt(e.cts, dk[j], y[j])

			if err != nil {
				fmt.Println("error at i :", j)
				log.Fatal(err)
			}
			helper, _ := res.Float64()
			theta[j] = helper / float64(e.a.scaling*e.a.scaling)
			//fmt.Printf("theta[%v]: %v\n", j, theta[j])

		}

		//fmt.Printf("theta: %v\n", theta)

		timeI := time.Since(startI)
		if i%10 == 0 {
			fmt.Println("it ", i)
		}

		//debug(fmt.Sprintf("time Iteration %v: %v\n",i, timeI))
		UNUSED(timeI, tDK)
	}
	//	fmt.Printf("eI: %v\n", eI)

	timeGD := time.Since(start)
	//fmt.Printf("time whole GD: %v\n", timeGD)

	return theta, timeGD, nil

}

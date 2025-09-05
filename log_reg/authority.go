package main

import (
	"fmt"
	"log"
	"math"
	"math/big"
	"time"

	"github.com/JasZal/gofe/data"
	"github.com/JasZal/gofe/innerprod/noisy"
	"github.com/fentec-project/bn256"
	"github.com/google/differential-privacy/go/noise"
)

var noisyB bool

type Authority struct {
	vecLen    int
	numClient int
	boundX    *big.Int
	boundY    *big.Int
	boundN    *big.Int
	pubKey    *bn256.GT
	msk       [][]byte
	fe        *noisy.OTPRF
	epsilon   float64
	delta     float64
	scaling   int64
	nrWorkers int
	keys      map[string]int
}

// Authority for the PPML training
// generates decryption keys and keeps track of the privacy budget and computes noise
func NewAuthority(vecL int, numC int, bX, bY, bN *big.Int, e, d float64, scal int64, keys map[string]int, workers int, n bool) (*Authority, time.Duration) {
	a := &Authority{
		vecLen:    vecL,
		numClient: numC,
		boundX:    bX,
		boundY:    bY,
		boundN:    bN,
		epsilon:   e,
		delta:     d,
		scaling:   scal,
		nrWorkers: workers,
		keys:      keys,
	}
	noisyB = n
	start := time.Now()
	a.fe = noisy.NewOTPRF(a.numClient, a.vecLen, a.boundX, a.boundY, a.boundN)
	timeSetup := time.Since(start)
	a.msk = a.fe.GenerateKeys()

	return a, timeSetup
}

// computes inf senf for log reg
func computeInfSen(theta []float64, alpha, n float64) float64 {
	sumT := 0.0
	for i := 0; i < len(theta); i++ {
		sumT += math.Abs(theta[i])
	}

	a3 := -0.81562 / 512
	a1 := -1.20086 / 8

	return alpha / n * (1 + math.Abs(a3*math.Pow(sumT, 3)-a1*sumT))

}

// computes zero sen. for log reg.
func compute0Sen(theta []float64) int64 {
	return int64(len(theta))
}

// computes gaussian noise to perturb decryption keys
func computeNoise(theta []float64, alpha, b, scaling, eps, del float64) []float64 {
	n := make([]float64, len(theta))

	infSen := computeInfSen(theta, alpha, b)
	zeroSen := compute0Sen(theta)

	//noise via gauss
	gauss := noise.Gaussian()
	for j := 0; j < len(n); j++ {
		n[j] = gauss.AddNoiseFloat64(0.0, zeroSen, infSen, eps, del)

	}

	//scale noise
	for j := 0; j < len(n); j++ {
		n[j] *= math.Pow(scaling, 2)

		if n[j] >= 0 {
			n[j] = math.Ceil(n[j])
		} else {
			n[j] = math.Floor(n[j])
		}
	}

	if !noisyB {
		n = make([]float64, len(theta))
	}

	return n
}

// function to derive a functional key for vector y
func (a *Authority) generateFunctionKey(y data.Matrix, noise *big.Int, label []byte) (*big.Int, error) {

	key, err := a.fe.DeriveKey(a.msk, y, noise, label, a.nrWorkers)
	if err != nil {
		fmt.Println("Error during key derivation:", err)
		return nil, err
	}

	return key, nil
}

// function that generates decryption key for the log reg update function with respect to given
// model weights theta,
// //privacy budget eps, del
// learning rate alpha
func (a Authority) generateDK(theta []float64, m int, batch []int, eps, del, alpha float64) ([]*big.Int, []data.Matrix) {
	dk := make([]*big.Int, m+1)
	y := make([]data.Matrix, m+1)

	//approx sigmoid:
	a3 := -0.81562 / 512
	a1 := -1.20096 / 8
	a0 := 0.5

	//2^3m
	pow23m := new(big.Int).Exp(big.NewInt(8), big.NewInt(int64(m)), nil) // 	 int(math.Pow(2, 3*float64(m)))

	//bias term, (theta[m+1])
	y[m] = data.NewConstantMatrix(a.numClient, a.vecLen, big.NewInt(0))

	//constant term
	yH := theta[m] - alpha*(a0-a1*theta[m]+a3*math.Pow(theta[m], 3))
	y[m][0][0] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))

	for _, i := range batch {
		//for i := 0; i < a.numClient; i++ {
		//terms of degree 1
		//coeff for x[m+1] -- lookup index in keys
		yH = alpha / float64(a.numClient)
		y[m][i][a.keys[new(big.Int).Add(pow23m, big.NewInt(int64(m+1))).String()]] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))

		for k := 0; k < m; k++ {
			//x_i[k]  -- lookup index in keys for 8^i
			yH = (alpha * theta[k] / float64(a.numClient)) * (a1 - 3*a3*math.Pow(theta[m], 2))
			y[m][i][a.keys[new(big.Int).Exp(big.NewInt(8), big.NewInt(int64(k)), nil).String()]] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))
		}

		//multinomial coefficients
		var generate func(m int, d int, index int, currentSum int, coeff float64, ks []int, key *big.Int)
		generate = func(m int, d int, index int, currentSum int, coeff float64, ks []int, key *big.Int) {
			if index == m {
				if currentSum == d {
					// Calculate the  coefficient
					yH := float64(MultinomialCoefficient(d, ks)) * coeff
					for k := 0; k < m; k++ {
						yH *= math.Pow(theta[k], float64(ks[k]))
					}

					//fmt.Printf("coeff: %v, exp: %v, key: %v\n", yH, ks, key)
					y[m][i][a.keys[key.String()]] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))
					key = big.NewInt(0)
				}
				return
			}

			for i := 0; i <= d-currentSum; i++ {
				ks[index] = i
				if i != 0 {
					key = new(big.Int).Add(key, new(big.Int).Exp(big.NewInt(8), big.NewInt(int64(index)), nil))
				}

				generate(m, d, index+1, currentSum+i, coeff, ks, key)
			}
		}

		//terms of degree 2
		coeff := (alpha / float64(a.numClient)) * (-3 * a3 * theta[m])
		ks := make([]int, m)
		generate(m, 2, 0, 0, coeff, ks, big.NewInt(0))

		//terms of degree 3
		coeff = (-1 * alpha * a3) / float64(a.numClient)
		ks = make([]int, m)
		generate(m, 3, 0, 0, coeff, ks, big.NewInt(0))

	}

	for j := 0; j < m; j++ {
		y[j] = data.NewConstantMatrix(a.numClient, a.vecLen, big.NewInt(0))

		//constant term
		yH := theta[j]
		y[j][0][0] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))

		for _, i := range batch {
			yH = alpha / float64(a.numClient) * (-1*a0 + a1*theta[m] - a3*math.Pow(theta[m], 3))
			y[j][i][a.keys[new(big.Int).Exp(big.NewInt(8), big.NewInt(int64(j)), nil).String()]] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))

			//xi[m+1]xi[j]
			yH = alpha / float64(a.numClient)
			y[j][i][a.keys[new(big.Int).Add(pow23m, big.NewInt(int64(j))).String()]] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))

			//xi[k]xi[j]
			for k := 0; k < m; k++ {
				yH = alpha * theta[k] / float64(a.numClient) * (a1 - 3*a3*math.Pow(theta[m], 2))

				y[j][i][a.keys[new(big.Int).Add(new(big.Int).Exp(big.NewInt(8), big.NewInt(int64(j)), nil), new(big.Int).Exp(big.NewInt(8), big.NewInt(int64(k)), nil)).String()]] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))

			}

			//multinomial coefficients
			var generate func(m int, d int, index int, currentSum int, coeff float64, ks []int, key *big.Int)
			generate = func(m int, d int, index int, currentSum int, coeff float64, ks []int, key *big.Int) {
				if index == m {
					if currentSum == d {
						// Calculate the coefficient
						yH := float64(MultinomialCoefficient(d, ks)) * coeff
						for k := 0; k < m; k++ {
							yH *= math.Pow(theta[k], float64(ks[k]))
						}

						key = new(big.Int).Add(key, new(big.Int).Exp(big.NewInt(8), big.NewInt(int64(j)), nil))
						y[j][i][a.keys[key.String()]] = big.NewInt(int64(math.Round(yH * float64(a.scaling))))
						key = big.NewInt(0)
					}
					return
				}

				for i := 0; i <= d-currentSum; i++ {
					ks[index] = i
					if i != 0 {
						key = new(big.Int).Add(key, new(big.Int).Exp(big.NewInt(8), big.NewInt(int64(index)), nil))

					}

					generate(m, d, index+1, currentSum+i, coeff, ks, key)
				}
			}

			//degree 3
			coeff := alpha / float64(a.numClient) * (-3 * a3 * theta[m])
			ks := make([]int, m)
			generate(m, 2, 0, 0, coeff, ks, big.NewInt(0))

			//degree 4
			coeff = -1 * alpha * a3 / float64(a.numClient)
			ks = make([]int, m)
			generate(m, 3, 0, 0, coeff, ks, big.NewInt(0))

		}

	}

	var err error

	label := make([]byte, 16)
	noise := computeNoise(theta, alpha, float64(len(batch)), float64(a.scaling), eps, del)
	for k := 0; k < m+1; k++ {
		dk[k], err = a.generateFunctionKey(y[k], big.NewInt(int64(noise[k])), label)
	}
	if err != nil {
		log.Fatal(err)
	}

	return dk, y
}

func (a Authority) getEncryptionKey(pos int) []byte {
	return a.msk[pos]
}

func (a Authority) getParams() *noisy.OTPRFParams {
	return a.fe.Params
}

func Factorial(n int) int {
	if n == 0 || n == 1 {
		return 1
	}
	result := 1
	for i := 2; i <= n; i++ {
		result *= i
	}
	return result
}

// MultinomialCoefficient calculates the multinomial coefficient d! / (k1! * k2! * ... * km!)
func MultinomialCoefficient(d int, ks []int) int {
	denominator := 1
	for _, k := range ks {
		denominator *= Factorial(k)
	}
	return Factorial(d) / denominator
}

package utils

import "math/rand"

// from `1` to `n`
func RandInt(n uint) int {
	return rand.Intn(int(n)) + 1
}

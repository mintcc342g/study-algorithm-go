package utils

import "math/rand"

// 1 ~ n
func RandInt(n uint) int {
	return rand.Intn(int(n)) + 1
}

// from ~ to
func RandIntRange(from, to uint) int {
	if from > to {
		return int(to) + rand.Intn(int(from-to)+1)
	}
	return int(from) + rand.Intn(int(to-from)+1)
}

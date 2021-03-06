package array

import (
	"errors"
	"fmt"
)

/*
 ** 정렬된 배열에서의 이진 검색
 */
func binarySearch(sortedArray []int, value int) error {

	lower := 0
	upper := len(sortedArray) - 1
	var mid int
	var midPoint int

	for lower <= upper {
		mid = (lower + upper) / 2   // 우선 절반 나눠서
		midPoint = sortedArray[mid] // 가운데 값 구함.

		if value < midPoint { // 가운데 값 보다 찾으려는 값이 작으면,
			upper = mid - 1 // 최대값 범위를 하나 낮춰줌.
		} else if midPoint < value { // 가운데 값 보다 찾으려는 값이 크면,
			lower = mid + 1 // 최소값 범위를 하나 높여줌.
		} else if value == midPoint { // 일치하면 끝!
			return nil
		}
	}

	return errors.New("not found")
}

/*
 *** 버블 정렬
 */
func bubbleSort(l []int) []int {
	println("Before bubble sort", fmt.Sprintf("%v", l))

	sortedIdx := len(l) - 1 // 정렬 완료된 부분은 빼고 for문 돌기 위한 카운트
	sorted := false         // 전체 정렬 완료 여부

	for !sorted {
		sorted = true

		for i := 0; i < sortedIdx; i++ {
			if l[i] > l[i+1] { // 부등호가 > 이거라서 asc 정렬이 되는데, < 이거면 desc 정렬 됨.
				sorted = false
				l[i], l[i+1] = l[i+1], l[i]
			}
		}
		sortedIdx = sortedIdx - 1 // 매 passthrough 마다 가장 큰 값은 정렬이 됐을거니까 -1씩 해주는 것
	}

	println("After bubble sort", fmt.Sprintf("%v", l))

	return l
}

/*
 *** 선택 정렬
 */
func selectionSort(l []int) {
	println("\nBefore selection sort", fmt.Sprintf("%v", l))

	var minValIdx int
	for i, _ := range l {
		minValIdx = i
		for k := i + 1; k < len(l); k++ { // i번째 다음부터 진행하는 이유는 i번째를 포함해서 그 앞은 정렬이 되어 있으니까
			if l[k] < l[minValIdx] {
				minValIdx = k
			}
		}

		if minValIdx != i { // 최소값이 갱신되었을 경우 배열값도 스와프 해줌.
			l[i], l[minValIdx] = l[minValIdx], l[i]
		}
	}

	println("After selection sort", fmt.Sprintf("%v", l))
}

/*
 *** 삽입 정렬
 */
func insertionSort(l []int) []int {
	println("\nBefore insert sort", fmt.Sprintf("%v", l))

	var position int
	var temp int

	for i := 1; i < len(l); i++ {
		position = i
		temp = l[i]

		for position != 0 && l[position-1] > temp {
			l[position] = l[position-1]
			position = position - 1
		}

		l[position] = temp
	}

	println("After insert sort", fmt.Sprintf("%v", l))
	return l
}

// leetcode
func bitXOR(nums []int) {
	// 짝수번 중복됐을 때에만 제외해줌.
	// 중복되지 않은 수가 2개 이상이면, 둘을 비트 계산해버려서 숫자가 달라짐.
	// 중복 순서는 상관없음.
	res := nums[0]
	for i := 1; i < len(nums); i++ {
		res = res ^ nums[i]
	}

	println("\nbit XOR :", res)
}

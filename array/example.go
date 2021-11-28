package array

import "errors"

// 정렬된 배열에서의 이진 검색
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

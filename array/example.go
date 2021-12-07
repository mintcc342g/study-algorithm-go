package array

import (
	"errors"
	"fmt"
)

/*
 *** 정렬된 배열에서의 이진 검색
 * 이진 검색은 전체 데이터를 절반씩 나누면서 검색하는 방법
 * 예를 들어, 1~100 의 숫자 중 3 을 찾는다고 해보자.
 * 우선 1~100의 중간값인 50을 뽑고,
 * 3이 50보다 크다면 범위를 51~100 으로 잡음,
 * 3이 50보다 작다면 범위를 1~50 으로 잡음.
 * 위의 과정을 반복 (passthrough)

 * 범위를 절반씩 좁혀가기 때문에 O(log N)의 효율을 가짐.
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
 * 버블 정렬은 배열 내 연속된 두 항목을 비교하며 정렬하는 방식
 * 비교하는 2개의 항목 중 가장 큰 값(=버블)을 순서에 맞는 위치로
 * 옮기기 때문에 버블 정렬이라고 함.

 * 비교 후 스와프까지 하기 때문에 한 passthrough 당 2단계가 걸림.
 * 그래서 효율성은 O(N^2) 임.
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
 * 선택 정렬은 배열의 각 셀을 돌며 셀의 값과 최소값을 비교하며 정렬하는 방식
 * 예를 들어, 시작은 0번째 인덱스부터 하고 최소값 설정도 0번 인덱스의 값부터.
 * 1번 인덱스의 값고 0번째를 비교
 * 1번 인덱스 값이 더 크면 0번 인덱스 값을 최소값으로 유지
 * 0번 인덱스 값이 더 크면 1번 인덱스 값을 최소값으로 변경
 * 위 과정을 반복

 * 선택 정렬은 모든 시나리오에서 O(N^2) 효율을 가짐.
 * 이미 정렬된 부분은 검사하지 않고 건너 뛰기 때문인 듯

 * 사실 버블 정렬보다 2배 더 효율이 좋으므로 O(N^2 / 2) 라고
 * 표시해야 할 것 같지만, 빅 오 표기법에서는 부가적인 정보는
 * 빼기 때문에 / 2 를 빼서 O(N^2) 로 표기함.
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
 * 최소값과 현재 배열의 값을 비교한 후, 최소값을 배열에 밀어넣으며 정렬을 하는 방식
 * 상세 과정은 다음과 같음.
 * 시작할 때는 1번 인덱스의 값을 최소값으로 설정한 후 시작
 * 최소값과 모든 배열의 값을 비교함.
 * 만약 현재 배열의 값이 최소값보다 더 크다면, 현재 배열의 위치에 최소값을 넣어줌.
 * 그리고 현재 배열의 값을 최소값 자리로 옮김.
 * 이상을 반복

 * 삽입 정렬의 효율은 시나리오에 따라 많이 달라짐.
 * 최악 : N^2 단계
 * 평균 : N^2 / 2 단계
 * 최선 : N 단계
 */
func insertionSort(l []int) []int {
	println("\nBefore insert sort", fmt.Sprintf("%v", l))

	var position int
	var temp int

	for i := 1; i < len(l); i++ {
		position = i
		temp = l[i]

		for l[position-1] > temp {
			l[position] = l[position-1]
			position = position - 1
		}

		l[position] = temp
	}

	println("After insert sort", fmt.Sprintf("%v", l))
	return l
}

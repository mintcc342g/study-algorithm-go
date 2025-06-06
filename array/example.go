package array

import (
	"errors"
	"fmt"
)

/*
 ** 정렬된 배열에서의 이진 검색
 */
func BinarySearch(sortedArray []int, value int) (int, error) {
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
			return mid, nil
		}
	}

	return -1, errors.New("not found")
}

// leetcode
func searchInsert(nums []int, target int) int {
	// nums 내에서 target 을 찾아서 그 인덱스를 반환해야 함.
	// 만약 찾지 못했을 경우엔 target을 삽입할 적절한 인덱스를 반환해야 함.
	// 제약조건이 O(log n) 이므로, 이진 검색 해야됨.

	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}

	// for문 종료조건이 left > right가 됐을 때임.
	// 즉, for문이 종료됐을 때 right는 항상 left - 1 이 됨.
	// for문을 나왔다는 건 target이 삽입되어야 하는 위치를 찾아야 한다는 거고,
	// left == right 인 경우가 바로 삽입해야 하는 위치임.
	// 따라서 left를 반환하는 것
	return left
}

/*
 ** 버블 정렬
 */
func BubbleSort(l []int) []int {
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

	return l
}

/*
 ** 선택 정렬
 */
func SelectionSort(l []int) []int {
	var minValIdx int
	for i := range l {
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

	return l
}

/*
 ** 삽입 정렬
 */
func InsertionSort(l []int) []int {
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

	return l
}

/*
 ** 퀵 정렬
 */

func quickSort(l []int, desc bool) []int {
	return qsort(l, 0, len(l)-1, desc)
}

func qsort(l []int, left, right int, desc bool) []int {
	if right-left < 1 {
		return l
	}

	var pivot int
	if desc {
		pivot = partitionDesc(l, left, right)
	} else {
		pivot = partitionAsc(l, left, right)
	}
	l = qsort(l, left, pivot-1, desc)
	l = qsort(l, pivot+1, right, desc)

	return l
}

func partitionAsc(l []int, left, right int) int {
	fmt.Printf("list: %v, left: %d, right: %d\n", l, left, right)
	pivot := right
	right--
	for left <= right {
		if l[left] <= l[pivot] {
			left++
		} else if l[right] >= l[pivot] {
			right--
		} else { // else if l[left] >= l[pivot] && l[right] <= l[pivot]
			l[left], l[right] = l[right], l[left]
		}
	}

	l[left], l[pivot] = l[pivot], l[left]

	return left
}

func partitionDesc(l []int, left, right int) int {
	fmt.Printf("list: %v, left: %d, right: %d\n", l, left, right)
	pivot := left
	left++
	for left <= right {
		if l[left] >= l[pivot] {
			left++
		} else if l[right] <= l[pivot] {
			right--
		} else { // else if l[left] >= l[pivot] && l[right] <= l[pivot]
			l[left], l[right] = l[right], l[left]
		}
	}

	l[right], l[pivot] = l[pivot], l[right]

	return right
}

/*
 ** 퀵 선택
 */

func QuickSelection(l []int, k uint, largest bool) int {
	// k번째는 1부터 시작하는데, 기준이 되는 pivot은 0부터 시작하니까 k-1 해줌.
	return qselection(l, int(k-1), largest, 0, len(l)-1)
}

func qselection(l []int, k int, largest bool, left, right int) int {
	if right-left < 1 {
		return l[right]
	}

	var pivot int
	if largest {
		pivot = partitionDesc(l, left, right)
	} else {
		pivot = partitionAsc(l, left, right)
	}

	if k > pivot {
		return qselection(l, k, largest, pivot+1, right)
	} else if k < pivot {
		return qselection(l, k, largest, left, pivot-1)
	}

	return l[pivot]
}

/*
 ** 병합 정렬
 * (연결리스트 예제도 있는데, linkedlists 쪽 확인)
 */
func mergeSort(arr []int) []int {
	// 배열 크기가 1 이하이면 이미 정렬된 상태
	if len(arr) <= 1 {
		return arr
	}

	// 배열을 절반으로 나누기
	mid := len(arr) / 2
	left := mergeSort(arr[:mid])
	right := mergeSort(arr[mid:])

	// 나뉜 배열 병합
	return merge(left, right)
}

func merge(left, right []int) []int {
	result := make([]int, 0, len(left)+len(right))
	i, j := 0, 0

	// 두 배열 비교하며 작은 값을 결과에 추가
	for i < len(left) && j < len(right) {
		if left[i] <= right[j] {
			result = append(result, left[i])
			i++
		} else {
			result = append(result, right[j])
			j++
		}
	}

	// 남은 요소를 결과에 추가
	result = append(result, left[i:]...)
	result = append(result, right[j:]...)

	return result
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

// 퀵 정렬, 퀵 셀렉트
type SortableArray struct {
	arr []int
}

func NewSortableArray(arr []int) *SortableArray {
	return &SortableArray{
		arr: arr,
	}
}

func (s *SortableArray) partition(leftPtr, rightPtr int) int {
	pivot := rightPtr // 항상 오른쪽에 있는 값을 피벗으로 선정
	pivotVal := s.arr[pivot]

	rightPtr = pivot - 1 // 피벗의 바로 왼쪽 인덱스가 오른쪽 포인터의 시작점

	for {
		for s.arr[leftPtr] < pivotVal { // 왼쪽 포인터의 이동
			leftPtr += 1
		}
		for s.arr[rightPtr] > pivotVal { // 오른쪽 포인터의 이동
			rightPtr -= 1
		}

		if leftPtr >= rightPtr { // 왼쪽 포인터와 오른쪽 포인터의 위치가 같거나,
			break // 왼쪽 포인터가 오른쪽 포인터의 위치를 넘어섰으면 분할 중지
		}

		s.swap(leftPtr, rightPtr) // 두 포인터의 이동이 멈췄다면, 두 포인터가 가리키는 값을 교환함.
	}

	// 분할 과정이 끝났다면, 왼쪽 포인터의 값과 피벗 값을 교환함.
	s.swap(leftPtr, pivot)

	return leftPtr // 이건 예제의 quickSort 메서드를 위해서 왼쪽 포인터 반환하는 것
}

func (s *SortableArray) swap(fstIdx, sndIdx int) {
	s.arr[fstIdx], s.arr[sndIdx] = s.arr[sndIdx], s.arr[fstIdx]
}

// 퀵 정렬을 재귀로 구현
func (s *SortableArray) quickSort(leftIdx, rightIdx int) {
	// 기저 조건은 하위 배열의 원소가 0~1개 일 때
	if rightIdx-leftIdx <= 0 {
		return
	}

	// 배열을 분할하고 피벗 위치를 가져옴.
	// partition 은 분할을 끝낸 배열의 왼쪽 포인터를 반환하는데,
	// 분할이 끝난 배열의 왼쪽 포인터는 해당 배열의 맨 오른쪽 값을 가리키고 있을 것이기 때문임.
	pivot := s.partition(leftIdx, rightIdx)

	// 피벗의 왼쪽 배열에 대한 퀵 정렬
	// 이 하위 배열의 오른쪽 포인터는 피벗의 한 칸 왼쪽에서 시작되어야 하므로 pivot -1
	s.quickSort(leftIdx, pivot-1)

	// 피벗의 오른쪽에 대한 퀵 정렬
	// 이 하위 배열의 왼쪽 포인터는 피벗의 한 칸 오른쪽에서 시작되어야 하므로 pivot + 1
	s.quickSort(pivot+1, rightIdx)
}

// 퀵 셀렉션 구현
// 정렬되지 않은 배열에서 n번째로 작은 값을 찾음. (n은 0부터 시작)
func (s *SortableArray) quickSelectLowest(nth, leftIdx, rightIdx int) int {
	// 기저 조건은 하위 배열의 원소가 1개가 됐을 때
	if rightIdx-leftIdx <= 0 {
		return s.arr[leftIdx]
	}

	// 배열을 분할하고 피벗 위치를 가져옴.
	// 퀵 셀렉션도 분할을 기반으로 하니까 똑같이 partition 이용해줌.
	pivot := s.partition(leftIdx, rightIdx)

	if nth < pivot { // 찾고자 하는 값의 순위가 피벗 기준 왼쪽에 있다면
		s.quickSelectLowest(nth, leftIdx, pivot-1) // 왼쪽 재귀 분할 시작
	} else if nth > pivot { // 오른쪽일 경우
		s.quickSelectLowest(nth, pivot+1, rightIdx) // 오른쪽 재귀 분할 시작
	}

	// 책에서는 찾고자 하는 값의 인덱스 == pivot 이라면서
	// s.arr[pivot] 하면 된다는데, 실제로 해보면 그렇지 않음.
	// 추측컨대, partion 을 한 번 더 해야 하는데 못해서 그러는 게 아닌가 싶기도..?
	return s.arr[nth]
}

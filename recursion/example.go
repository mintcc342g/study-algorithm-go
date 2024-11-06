package recursion

import "fmt"

// for문 보다 재귀가 더 빠르다고는 함.
// 단, 재귀적으로 정의된 경우에만 해당되며, 그렇지 않은 경우엔 오히려 효율이 떨어지므로 재귀는 적절한 해결법이 되지 못 함.

func Euclidean(x, y int) int {
	if y == 0 {
		return x
	}

	return Euclidean(y, x%y)
	// x 와 y 의 크기를 비교하지 않는 이유는, y 값을 x 자리에 넣어주기 때문.
	// 만약 첫 번째 재귀함수 인자로 x > y 로 들어온 경우, 두 번째 재귀함수 인자는 x' < y' 가 됨.
	// 예를 들어, x = 22, y = 8 로 넣었을 경우, 두번째 재귀함수 인자인 x' = 8 이 들어가고, y' = 22 가 들어가게 됨.
}

func TowerOfHanoi(no, from, to int) {
	// no: 원반 개수
	// from: 시작 기둥
	// to: 끝 기둥(목표 기둥)

	if no > 1 {
		TowerOfHanoi(no-1, from, 6-from-to)
		// 6 은 from 와 to 가 1과 3으로 들어왔을 때, 모든 기둥들(1,2,3) 의 합이라는 듯
		// 그럼 기둥이 몇 개 들어올지 모르는 경우엔..?
	}

	fmt.Printf("원반 %d 개를 %d 기둥에서 %d 기둥으로 옮깁니다.\n", no, from, to)

	if no > 1 {
		TowerOfHanoi(no-1, 6-from-to, to)
	}
}

// 일반적인 방식은 탑이 3개인 경우만 가정하고, from, via, to 라는 3개의 인자를 받음.
// 그리고 no - 1 을 반복함.
func HanoiNormal(no, from, via, to int) {
	if no == 1 {
		println(from, "->", to)

	} else {
		HanoiNormal(no-1, from, via, to)
		println(from, "->", to)
		HanoiNormal(no-1, via, to, from)
	}
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
func (s *SortableArray) QuickSort(leftIdx, rightIdx int) {
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
	s.QuickSort(leftIdx, pivot-1)

	// 피벗의 오른쪽에 대한 퀵 정렬
	// 이 하위 배열의 왼쪽 포인터는 피벗의 한 칸 오른쪽에서 시작되어야 하므로 pivot + 1
	s.QuickSort(pivot+1, rightIdx)
}

// 퀵 셀렉션 구현
// 정렬되지 않은 배열에서 n번째로 작은 값을 찾음. (n은 0부터 시작)
func (s *SortableArray) QuickSelectLowest(nth, leftIdx, rightIdx int) int {
	// 기저 조건은 하위 배열의 원소가 1개가 됐을 때
	if rightIdx-leftIdx <= 0 {
		return s.arr[leftIdx]
	}

	// 배열을 분할하고 피벗 위치를 가져옴.
	// 퀵 셀렉션도 분할을 기반으로 하니까 똑같이 partition 이용해줌.
	pivot := s.partition(leftIdx, rightIdx)

	if nth < pivot { // 찾고자 하는 값의 순위가 피벗 기준 왼쪽에 있다면
		s.QuickSelectLowest(nth, leftIdx, pivot-1) // 왼쪽 재귀 분할 시작
	} else if nth > pivot { // 오른쪽일 경우
		s.QuickSelectLowest(nth, pivot+1, rightIdx) // 오른쪽 재귀 분할 시작
	}

	// 책에서는 찾고자 하는 값의 인덱스 == pivot 이라면서
	// s.arr[pivot] 하면 된다는데, 실제로 해보면 그렇지 않음.
	// 추측컨대, partion 을 한 번 더 해야 하는데 못해서 그러는 게 아닌가 싶기도..?
	return s.arr[nth]
}

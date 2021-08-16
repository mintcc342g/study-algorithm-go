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

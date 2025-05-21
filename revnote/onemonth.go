package revnote

import (
	"math"
	"sort"
)

/*
 **
 * Subarray Division 1
 * 초콜릿 바를 어쩌고.. 하는 문제
 * s의 서브 배열의 길이가 m으로 고정된 상태에서
 * 서브 배열의 합이 d가 나오는 경우의 수를 구하는 문제
 * fixed-size의 슬라이딩 윈도우
 */
func birthday(s []int32, d int32, m int32) int32 {
	// sliding window
	var sum, cnt int32
	for i, v := range s {
		if i < int(m) { // m 크기의 윈도우가 만들어질 때까지
			sum += v                       // 합을 그냥 더해줌
			if i == int(m)-1 && sum == d { // m 크기의 윈도우가 만들어진 순간
				cnt++ // 만약 합이 d가 됐다면 카운팅
			}
		} else { // m 크기의 윈도우가 만들어진 이후
			sum = sum - s[i-int(m)] + s[i]
			if sum == d { // 이땐 합만 챙기면 됨.
				cnt++
			}
		}

	}

	return cnt
}

/*
 **
 * Max Min
 * 주어진 배열에서 요소를 k개 뽑아서 하위 배열을 만듦.
 * 그 하위 배열은 unfairness가 작아야 함 = 하위 배열을
 * 구성하는 요소들 간에 차이가 작아야 한다는 뜻.
 * 그 하위 배열의 최대값 - 최소값을 리턴하는 문제.
 * 하위 배열을 만들 때 arr의 요소를 사용하기만 하면 되므로,
 * arr을 정렬 후 크기가 k인 윈도우를 움직이면서 문제를 풀어야 함.
 * 즉, fixed-size의 슬라이딩 윈도우 문제임.
 */
func maxMin(k int32, arr []int32) int32 {
	size := int32(len(arr))
	if size == 0 || k > size || k <= 0 {
		return 0
	}

	// arr에서 아무 요소나 뽑아서 하위 배열을 만들면 되므로
	// 그냥 오름차순 정렬을 시켜버리면 됨.
	sort.Slice(arr, func(i, j int) bool {
		return arr[i] < arr[j]
	})

	min := func(a, b int32) int32 {
		if a < b {
			return a
		}
		return b
	}

	var res int32 = math.MaxInt32
	for i := int32(0); i <= size-k; i++ {
		currMin := arr[i+k-1] - arr[i] // 사실상 크기가 k인 윈도우의 max값 - min값임. 또, unfairness한 하위 배열은
		res = min(currMin, res)        // max - min 값이 가장 작을 것이기 때문에, 그냥 윈도우 옮기면서 그 차이값을 구해주면 됨.
	}

	return res
}

/*
 **
 * Permuting Two Arrays
 * 배열 A랑 B의 값을 재배열(permute)해서 새로운 배열 A'와 B'를 만듦.
 * A'[i] + B'[i] >= k 를 만족하면 YES, 아니면 NO를 리턴
 * 여기서 i는 i번째가 아니라, 0부터 i까지임.
 * 즉, 재배열한 모든 배열의 합이 전부 k를 넘어야 하는 것
 */
func twoArrays(k int32, A []int32, B []int32) string {
	// Write your code here
	// 확률적으로 한 쪽 배열의 큰 수와 다른 한 쪽 배열의 작은 수를 더하는 편이 더 나을 것
	// 따라서, A는 오름차순 B는 내림차순 이런 식으로 정렬을 해서 조건 줘서 찾으면 됨.

	// sort 이용할 경우
	a := make([]int, len(A))
	for i, v := range A {
		a[i] = int(v)
	}
	b := make([]int, len(B))
	for i, v := range B {
		b[i] = int(v)
	}

	sort.Ints(a)
	sort.Sort(sort.Reverse(sort.IntSlice(b)))

	for i := 0; i < len(a); i++ {
		if a[i]+b[i] < int(k) {
			return "NO"
		}
	}

	return "YES"
}

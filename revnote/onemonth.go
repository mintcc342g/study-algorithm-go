package revnote

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

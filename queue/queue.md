# Queue

## Contents
- [Basic Concepts](#basic-concepts)

## Basic Concepts
- FIFO(First-In-First-Out); 선입선출 자료구조
- 스택에서 주로 구현하는 연산
  - `add`: item을 리스트의 끝부분에 추가
  - `remove`: 큐의 첫 번째 항목 제거
  - `peek`: 큐의 가장 위의 항목을 반환
  - `isEmpty`: 큐 비어있는 여부 확인
- 너비우선탐색(BFS)이나 캐시 구현할 때 종종 사용됨.
- 링 버퍼 큐
  - 일반적인 배열로 만드는 큐는 복잡도가 O(n) 인데, 선입선출을 지키려고 배열 내 원소 이동이 필요하기 때문
  - 링 버퍼 큐는 배열 내 원소를 옮길 필요 없이 front와 rear 의 값만 업데이트 하여 큐를 구현하므로 복잡도가 O(1)
  - 링 버퍼는 오래된 데이터는 버리는 용도로 사용할 수 있음.

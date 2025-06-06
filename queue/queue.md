# Queue

## Contents
- [Basic Concepts](#basic-concepts)
- [Ring Buffer](#ring-buffer)
- [Priority Queue](#priority-queue)

## Basic Concepts
- FIFO(First-In-First-Out); 선입선출 자료구조
- 큐에서 주로 구현하는 연산
  - `add`: item을 리스트의 끝부분에 추가
  - `remove`: 큐의 첫 번째 항목 제거
  - `peek`: 큐의 가장 위의 항목을 반환
  - `isEmpty`: 큐 비어있는 여부 확인
- 너비우선탐색(BFS)이나 캐시 구현할 때 종종 사용됨.

## Ring Buffer
  - 배열 내 원소 이동을 하지 않는 형태로 큐를 구현한 자료구조
  - ring 이라는 이름에서 알 수 있듯, 배열의 처음과 끝이 연결되어 있다고 봄.
  - front와 rear라는 포인터 값만 업데이트 하여 배열 요소를 관리함.
    - 선형 배열로 큐를 만들 경우 배열 내 원소 이동 때문에 enqueue/dequeue의 시간 복잡도가 O(n)인데 반해, 링버퍼는 포인터로 요소를 관리하므로 enqueue/dequeue의 시간 복잡도가 O(1)임.
  - 링 버퍼는 오래된 데이터는 버리는 용도로 사용할 수 있음.

## Priority Queue
- 각 요소가 우선순위를 가지는 큐
  - FIFO가 아니지만, '항목을 순차적으로 처리'한다는 추상적인 개념을 가지고 있어서 일단 큐라고 부른다는 듯?
- 삭제 시 가장 높은(또는 가장 낮은) 우선순위를 가진 요소가 먼저 삭제됨.
- 보통 Max-Priority Queue, Min-Priority Queue 로 나눔.
- 주요 연산
  - `insert` 또는 `enqueue`: 새로운 요소를 큐에 추가
  - `delete` 또는 `dequeue`: 우선순위가 가장 높은/낮은 요소를 제거
  - `peak`: 우선순위가 가장 높은/낮은 요소를 조회

### 우선순위 큐의 구현 방식
#### 배열, 연결 리스트
- 정렬되지 않았을 때
  - 시간 복잡도
    - 삽입: O(1)
    - 삭제: O(n)
    - 조회: O(n)
  - 공간 복잡도: O(n)
- 정렬됐을 때
  - 시간 복잡도
    - 삽입: O(n)
    - 삭제: O(1)
    - 조회: O(1)
  - 공간 복잡도: O(n)

#### heap
- 시간 복잡도
  - 삽입: O(log n)
  - 삭제: O(log n)
  - 조회: O(1)
    - 루트 노드가 항상 최소값/최대값이기 때문에 조회에 O(1) 밖에 안 걸림.
    - 특히 heap은 항상 균형을 유지하는 트리이기 때문에 성능면에서 BST보다 유리함.
- 공간 복잡도: O(n)

#### binary search tree
- 다른 큐는 안 되고, 우선순위 큐에서만 사용함.
- 시간 복잡도
  - 삽입: O(log n)
  - 삭제: O(log n)
  - 조회: O(log n)
    - BST는 `자식1<부모<자식2` 라는 특정한 순서를 만족해야 함.
    - 즉, 최소값은 맨 왼쪽 끝 노드, 최대값은 맨 오른쪽 끝 노드가 됨.
    - 따라서 힙과 다르게 조회에 O(log n)이 걸리는 것
    - 불균형한 경우엔 O(n)도 걸릴 수 있어서 heap보다 불리함. 
- 공간 복잡도: O(n)

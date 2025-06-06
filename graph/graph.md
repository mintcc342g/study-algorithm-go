# Graph

- 그래프란?
    - 데이터의 연결, 관계에 특화된 자료 구조
- 가중(치) 그래프; Weighted Graph란?
    - 그래프에 추가적인 정보(가중치)를 포함시킨 것

<br/>

## 용어 정리

- 정점; vertex, node
- 간선; edge
    - 노드를 연결한 선
- 인접한 정점; adjacent vertex
- degree
    - 무방향 그래프; undirected graph에서 vertex에 연결된 edge의 개수
- indegree / outdegree
    - directed graph에서 들어오고/나가는 방향을 나타내는 degree


## 관련 알고리즘
- BFS; Breadth-First Search
    - 그래프에서 최단 경로를 구할 때 큐와 함께 자주 사용함.
        - BFS는 레벨별로 탐색을 하기 때문에 최단 거리 구하기에 좋음.
    - 책에서는 example 에서 구현한 BFS의 시간 복잡도가 O(V + E) 라고 함.
        - V는 vertex를 방문하는 횟수
            - N이랑 같은데, N이라고 안 한 이유는 edge랑 구별하려고 다른 표기를 사용했다는 듯
        - E는 vertex에 연결된 edge를 통과하는 횟수
            - 사실 각 vertex는 큐에 추가될 때랑 vertex를 방문할 때 같은 edge 를 통과함.
            - 그래서 사실은 edge 당 2번씩 통과하므로 2E 가 맞는데, 빅오에서 상수는 무시하니까 E만 썼다는 듯
        - 그래프에서는 간선에 따라서도 달라지기 때문에 N 표기 안 하고 V, E 사용
- DFS; Depth-First Search
    - 검색 대상이 큰 문제나 경로의 가중치 등을 저장하는 문제에서 재귀/스택과 함께 자주 사용함.
    - DFS도 마찬가지로 시간 복잡도는 O(V + E)

<br/>

- 데이크스트라; Dijkstra 알고리즘
    - shortest path 문제의 가장 기본 방식이자 greedy algorithm 으로 분류됨.
    - edge 값이 0 보다 작지 않을 때에만 정상 작동
    - 단순 반복문만 사용하는 방식이나, Heap 자료구조를 기반으로 Priority Queue 를 사용하여 구현하는 방식 등이 있음.
        - 시간 복잡도가 전자는 O(V^2), 후자는 O(E log V)
- 플로이드 워셜; Floyd-Warshall 알고리즘
    - 경로 찾을 때, 특정 지점을 반드시 거쳐야 한다면 이걸 쓴다는 듯

<br/>

- Disjoint Set
    - 서로 중복되지 않는 부분 집합들로 나눠진 원소들의 자료구조
    - 용어
        - subset; 부분집합
            - A의 원소들이 B의 원소들 중 하나라도 겹치면, A는 B의 subset
        - superset; 초월집합
            - 위에서 B가 superset
        - mutually disjoint
            - 만약 A랑 B가 겹치는 원소가 하나도 없다면, A랑 B는 mutually disjoint 하다고 말함.
    - 연산
        - make_set(x)
            - x를 원소로 하는 새로운 set을 생성함.
        - union(x, y)
            - x가 속한 set과 y가 속한 set을 합침.
            - 합칠 땐 각 set의 루트 노드를 이어줌.
            - union by size
                - 두 트리를 합칠 때 원소수가 적은 트리를 원소 수가 많은 트리의 밑으로 넣는 방식
                - union by rank 도 비슷한데, 이 경우, 트리의 높이를 기준으로, 높이가 큰 트리 밑으로 높이가 작은 트리를 넣음.
            - union by height
                - 트리의 높이가 작은 set을 큰 셋의 서브트리로 합치는 방법
            - size든 height 든 find 연산에서 찾은 두 개의 root 값을 비교해서 더 큰 쪽으로 작은 쪽을 넣어서 합쳐주면 됨.
            - 이렇게 하는 이유는 두 집합을 합칠 때 효율적으로 트리의 높이를 줄이기 위해서임.
        - path compression
            - 모든 원소들이 루트 노드를 바라보게 하는 것
            - 예를 들어, 1*-2-3-4 그래프에서 루트가 1이라고 했을 때, 저런 그래프를 안 만들고 1*-2, 1*-3, 1*-4 이런 식으로 만들어주는 걸 말함.
            - path compression 을 하면 find를 더 효율적으로 할 수 있음.
        - find(x)
            - x가 속한 set의 root값을 찾음.
            - set이 만들어졌을 때 가장 먼저 들어간 원소가 보통 root가 되는 듯?
    - 참고
        - [Disjoint Set](https://ratsgo.github.io/data%20structure&algorithm/2017/11/12/disjointset/)

<br/>

- Topological Sort
    - 위상 정렬은 순서가 정해져있는 작업을 차례로 수행해야 할 때 사용
        - Directed acyclic graph(방향이 있고 순환하지 않는 그래프)에서 모든 노드를 방향성에 거스르지 않도록 순서대로 나열함.
        - 순환이 없을 경우에만 모든 노드에 방문하게 됨.
    - 큐나 스택을 이용하는데, 거기에 노드를 넣었던 순서가 위상 정렬을 수행한 결과가 됨.
    - 근데 간선으로 이어진 순서가 아니라, 그래프를 마치 수평으로 펼친 모양으로 나타낸다고 봐야함.
        - 그래서 이어지지 않은 노드 간에도 정렬을 해서 보여주고, 결과도 여러개가 나옴.
    - 개념
        - 진입차수(in-degree): 특정 노드로 들어오는 간선 개수
        - 진출차수(out-degree): 특정 노드에서 나가는 간선 개수
    - 구현 방식
        - 진입차수가 0인 노드를 큐에 넣음
        - 큐에서 노드를 꺼낸 후 그 노드의 진출차수를 줄임.
        - 큐가 빌 때까지 위 과정 반복
    - 사용하는 문제: 강의 등 선수 과정 표현, 컴파일 작업 순서, git같은 버전 히스토리 관리, 교착상태 탐지 등
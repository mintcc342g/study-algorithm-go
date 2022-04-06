# Graph

- 그래프란?
    - 데이터의 연결, 관계에 특화된 자료 구조

<br/>

- 용어 정리
    - 정점; vertex, node
    - 간선; edge
        - 노드를 연결한 선
    - 인접한 정점; adjacent vertex
    - degree
        - 무방향 그래프; undirected graph에서 vertext에 연결된 edge의 개수
    - indegree / outdegree
        - directed graph에서 들어오고/나가는 방향을 나타내는 degree
- 관련 알고리즘
    - BFS; Breadth-First Search, DFS; Depth-First Search
        - 책에서는 example 에서 구현한 BFS의 시간 복잡도가 O(V + E) 라고 함.
        - V는 vertex를 방문하는 횟수
            - N이랑 같은데, N이라고 안 한 이유는 edge랑 구별하려고 다른 표기를 사용했다는 듯
        - E는 vertex에 연결된 edge를 통과하는 횟수
            - 사실 각 vertex는 큐에 추가될 때랑 vertex를 방문할 때 같은 edge 를 통과함.
            - 그래서 사실은 edge 당 2번씩 통과하므로 2E 가 맞는데, 빅오에서 상수는 무시하니까 E만 썼다는 듯

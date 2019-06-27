package hungarian

import (
	"math"
)

/**
 * Author: Stanford
 * Date: Unknown
 * Source: Stanford Notebook
 * Description: Min cost bipartite matching. Negate costs for max cost.
 * Time: O(N^3)
 * Status: tested during ICPC 2015
 * Repo: https://github.com/lungsin/go-hungarian
 */

func zero(x float64) bool {
	return math.Abs(x) < 1e-10
}

// returns array of integer with size n and default value v.
func vi(n, v int) []int {
	ans := make([]int, n)
	for i := range ans {
		ans[i] = v
	}
	return ans
}

func fill(arr []int, val int) {
	for i := 0; i < len(arr); i++ {
		arr[i] = val
	}
}

// MinCostMatching computes Min cost bipartite matching. Negate costs for max cost.
// Return explanation:
// - L[i] is the index of the matched node for i-th node in the row, i.e node i from the row will be matched to node L[i] from the column.
// - R[i] is the index of the matched node for i-th node in the column, i.e node R[i] from the row will be matched to node i from the column.
// - ans is the cost of the matching.
func MinCostMatching(cost [][]float64) (value float64, L []int, R []int) {
	n := len(cost)
	mated := 0
	dist := make([]float64, n)
	u := make([]float64, n)
	v := make([]float64, n)
	dad := make([]int, n)
	seen := make([]int, n)

	// construct dual feasible soltuion
	for i := 0; i < n; i++ {
		u[i] = cost[i][0]
		for j := 1; j < n; j++ {
			u[i] = math.Min(u[i], cost[i][j])
		}
	}
	for j := 0; j < n; j++ {
		v[j] = cost[0][j] - u[0]
		for i := 1; i < n; i++ {
			v[j] = math.Min(v[j], cost[i][j]-u[i])
		}
	}

	/// find primal solution satisfying complementary slackness
	L = vi(n, -1)
	R = vi(n, -1)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if R[j] != -1 {
				continue
			}

			if zero(cost[i][j] - u[i] - v[j]) {
				L[i] = j
				R[j] = i
				mated++
				break
			}
		}
	}

	for ; mated < n; mated++ { // until solution is feasible
		s := 0
		for L[s] != -1 {
			s++
		}
		fill(dad, -1)
		fill(seen, 0)
		for k := 0; k < n; k++ {
			dist[k] = cost[s][k] - u[s] - v[k]
		}

		j := 0
		for { /// find closest
			j = -1
			for k := 0; k < n; k++ {
				if seen[k] != 0 {
					continue
				}
				if j == -1 || dist[k] < dist[j] {
					j = k
				}
			}

			seen[j] = 1
			i := R[j]
			if i == -1 {
				break
			}

			for k := 0; k < n; k++ {
				if seen[k] != 0 {
					continue
				}
				new_dist := dist[j] + cost[i][k] - u[i] - v[k]
				if dist[k] > new_dist {
					dist[k] = new_dist
					dad[k] = j
				}
			}
		}

		/// update dual variables
		for k := 0; k < n; k++ {
			if k == j || seen[k] != 0 {
				continue
			}
			w := dist[k] - dist[j]
			v[k] += w
			u[R[k]] -= w
		}
		u[s] += dist[j]

		/// augment along path
		for dad[j] >= 0 {
			d := dad[j]
			R[j] = R[d]
			L[R[j]] = j
			j = d
		}
		R[j] = s
		L[s] = j
	}

	value = float64(0)
	for i := 0; i < n; i++ {
		value += cost[i][L[i]]
	}
	return value, L, R
}

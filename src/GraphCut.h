#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>

using namespace cv;
/// class for graph cut algorithms on undirected graphs (a graph is represented by an adjacency matrix).
/** The GraphCutter class implements the CONTRACT/CAP algorithm from H. Nagamochi, T. Ono, T. Ibaraki, "Implementing an efficient
minimum capacity cut algorithm", Mathematical Programming 67 (1994). */
class GraphCut{

public:

typedef struct{
    std::vector<int> subset;
    int parent;
    std::vector<int> children;
    float cost;
}CutNode;

/// Applies a single minimum cut on a connected undirected graph
/** @param adjacencyMatrix the input symmetric adjacency matrix (0 for "no edge", >0 for edge capacity)
    @param subset1 pointer to the first subset result
    @param subset2 pointer to the second subset result
    @return the cut cost.
*/
static float minCut(Mat_<float> &adjacencyMatrix, std::vector<int> &subset1, std::vector<int> &subset2);

/// Applies minimum cut as long as a cut with costs smaller a given threshold exists and returns a list of subsets (for weighted graphs).
/** @param adjacencyMatrix the input symmetric adjacency matrix (0 for "no edge", >0 for edge capacity)
    @param threshold the maximum cut cost
    @return a list of subsets.
*/
static std::vector<std::vector<int> > thresholdCut(Mat_<float> &adjacencyMatrix, float threshold);

/// Applies minimum cut as long as a cut with costs smaller a given threshold exists and returns a list of subsets (unweighted graphs will be weighted).
/** @param adjacencyMatrix the input symmetric adjacency matrix (0 for "no edge", 1 for "edge")
    @param threshold the maximum cut cost
    @return a list of subsets.
*/
static std::vector<std::vector<int> > thresholdCut(Mat_<bool> &adjacencyMatrix, float threshold);

/// Applies hierarchical minimum cut and returns a list of nodes including subset, parent, children and weight representing a cut tree (for weighted graphs).
/** @param adjacencyMatrix the input symmetric adjacency matrix (0 for "no edge", >0 for edge weights)
    @return a list of tree-nodes.
*/
static std::vector<CutNode> hierarchicalCut(Mat_<float> &adjacencyMatrix);

/// Applies hierarchical minimum cut and returns a list of nodes including subset, parent, children and weight representing a cut tree (unweighted graphs will be weighted).
/** @param adjacencyMatrix the input symmetric adjacency matrix (0 for "no edge", 1 for "edge")
    @return a list of tree-nodes.
*/
static std::vector<CutNode> hierarchicalCut(Mat_<bool> &adjacencyMatrix);

/// Creates a weighted matrix from an unweighted boolean matrix.
/** @param initialMatrix the unweighted boolean matrix (0 for "no edge", 1 for "edge")
    @return a weighted matrix.
*/
static Mat_<float> calculateProbabilityMatrix(Mat_<bool> &initialMatrix, bool symmetry=true);

/// Find all unconnected subgraphs and return a list of the subsets
/** @param adjacencyMatrix the input matrix (0 for "no edge", >0 for edge weights)
    @return a list of subsets.
*/
static std::vector<std::vector<int> > findUnconnectedSubgraphs(Mat_<float> &adjacencyMatrix);

    /// Creates a submatrix with a given subset.
/** @param adjacencyMatrix the input matrix
    @param subgraph the subset of nodes in the matrix
    @return a matrix from the subset.
*/
    static Mat_<float> createSubMatrix(Mat_<float> &adjacencyMatrix, std::vector<int> &subgraph);

    /// Merges the src matrix into the dst matrix (dst is the maximum of both matrices).
/** @param dst the destination matrix
    @param src the source matrix
*/
    static void mergeMatrix(Mat_<bool> &dst, Mat_<bool> &src);

    /// Weights the dst matrix (value*=weight) for all true cells in the featureMatrix.
/** @param dst the destination matrix
    @param featureMatrix the boolean feature matrix (indicates where to weight)
    @param weight the weight
*/
    static void weightMatrix(Mat_<float> &dst, Mat_<bool> &featureMatrix, float weight);

private:
static std::vector<float> capforest(std::vector<Point> &edgeList, std::vector<float> &edgeCosts, int subsetsSize);

static float initialLambda(Mat_<float> &adjacencyMatrix, int &lambda_id);

static void createEdgeList(Mat_<float> &adjacencyMatrix, std::vector<Point> &edgeList, std::vector<float> &edgeCosts);

static std::vector<std::vector<int> > createInitialNodes(Mat_<float> &adjacencyMatrix);

static float merge(std::vector<Point> &edgeList, std::vector<float> &edgeCosts, std::vector<float> &q,
                                std::vector<std::vector<int> > &subsets, float lambda_score, int j, int &lambda_id);
};

#include <GraphCut.h>

float GraphCut::minCut(Mat &adjacencyMatrix, std::vector<int> &subset1, std::vector<int> &subset2){
  //Please note: it is possible to add an additional adjacency matrix for faster lookup with values pointing to the edgeList IDs.

  //Find minimal cut cost for a single node (initial lambda)
  int lambda_id = -1;
  float lambda_score = initialLambda(adjacencyMatrix, lambda_id);

  //create edge list and edge costs
  std::vector<Point> edgeList;
  std::vector<float> edgeCosts;
  createEdgeList(adjacencyMatrix, edgeList, edgeCosts);

  //initial single nodes
  std::vector<std::vector<int> > subsets = createInitialNodes(adjacencyMatrix);

  //merge to a graph with 2 nodes
  bool noQFound=false;
  while(subsets.size()>2 && noQFound==false){
    //calculate lower bounds
    std::vector<float> q = capforest(edgeList, edgeCosts, subsets.size());

    noQFound=true;
    for(unsigned int j=0; j<q.size(); j++){ //find q > initial cost of single node
      if((q[j]>lambda_score)||(q[j]>=lambda_score && edgeList[j].x!=lambda_id && edgeList[j].y!=lambda_id)){//the two nodes can be merged
        noQFound=false;
        //merge
        lambda_score = merge(edgeList, edgeCosts, q, subsets, lambda_score, j, lambda_id);
        j=0;//start searching q from beginning (due to q-updates for merging)
      }
    }
  }
  if(subsets.size()!=2){
    //lambda_id is already min-cut
    subset1 = subsets[lambda_id];
    subset2.clear();
    for(unsigned int i=0; i<subsets.size(); i++){
      if((int)i!=lambda_id){
        for(unsigned int j=0; j<subsets[i].size(); j++){
          subset2.push_back(subsets[i][j]);
        }
      }
    }
    return lambda_score;
  }
  subset1 = subsets[0];
  subset2 = subsets[1];
  return edgeCosts[0];
}


std::vector<std::vector<int> > GraphCut::thresholdCut(Mat &adjacencyMatrix, float threshold){
  Mat probabilities = calculateProbabilityMatrix(adjacencyMatrix, true);
  std::vector<std::vector<int> > subgraphs = findUnconnectedSubgraphs(probabilities);
  std::vector<std::vector<int> > children(subgraphs.size());//children IDs (empty = none)
  for(unsigned int i=0; i<subgraphs.size(); i++){
    if(subgraphs[i].size()>1){
      Mat_<float> subMatrix = createSubMatrix(probabilities, subgraphs[i]);

      std::vector<int> subset1;
      std::vector<int> subset2;
      float cost = GraphCut::minCut(subMatrix, subset1, subset2);
      if(cost<threshold){//add to list if cost smaller threshold
        std::vector<int> mappedSet1;
        std::vector<int> mappedSet2;
        for(unsigned int j=0; j<subset1.size(); j++){//map nodes to original input nodes
          mappedSet1.push_back(subgraphs[i][subset1[j]]);
        }
        for(unsigned int j=0; j<subset2.size(); j++){
          mappedSet2.push_back(subgraphs[i][subset2[j]]);
        }
        subgraphs.push_back(mappedSet1);
        subgraphs.push_back(mappedSet2);
        children.resize(subgraphs.size());
        std::vector<int> c;
        c.push_back(subgraphs.size()-2);
        c.push_back(subgraphs.size()-1);
        children[i]=c;
      }
    }
  }
  for(unsigned int i=0; i<children.size(); i++){//return only children
    if(children[i].size()>0){
      subgraphs.erase(subgraphs.begin()+i);
      children.erase(children.begin()+i);
      i--;
    }
  }
  return subgraphs;
}


std::vector<GraphCut::CutNode> GraphCut::hierarchicalCut(Mat &adjacencyMatrix){
  Mat probabilities = calculateProbabilityMatrix(adjacencyMatrix, true);
  std::vector<std::vector<int> > subsets = findUnconnectedSubgraphs(probabilities);
  std::vector<CutNode> cutNodes(subsets.size());
  for(unsigned int i=0; i<cutNodes.size(); i++){
    cutNodes[i].subset = subsets[i];
    cutNodes[i].parent=-1;//no parents for initial root nodes
  }
  for(unsigned int i=0; i<cutNodes.size(); i++){//process
    if(cutNodes[i].subset.size()>1){//inner nodes
      Mat_<float> subMatrix = createSubMatrix(probabilities, cutNodes[i].subset);
      std::vector<int> subset1;
      std::vector<int> subset2;
      float cost = GraphCut::minCut(subMatrix, subset1, subset2);

      std::vector<int> mappedSet1;
      std::vector<int> mappedSet2;
      for(unsigned int j=0; j<subset1.size(); j++){//map nodes to original input nodes
        mappedSet1.push_back(cutNodes[i].subset[subset1[j]]);
      }
      for(unsigned int j=0; j<subset2.size(); j++){
        mappedSet2.push_back(cutNodes[i].subset[subset2[j]]);
      }
      cutNodes.resize(cutNodes.size()+2);
      cutNodes[cutNodes.size()-2].subset=mappedSet1;//set subset for new nodes
      cutNodes[cutNodes.size()-1].subset=mappedSet2;
      cutNodes[cutNodes.size()-2].parent=i;//set parents for new nodes
      cutNodes[cutNodes.size()-1].parent=i;
      std::vector<int> c;
      c.push_back(cutNodes.size()-2);
      c.push_back(cutNodes.size()-1);
      cutNodes[i].children=c;//set new nodes as children
      cutNodes[i].cost=cost;//set costs in parent
    }else{//leaf nodes
      cutNodes[i].cost=0;
    }
  }
  return cutNodes;
}

std::vector<float> GraphCut::capforest(std::vector<Point> &edgeList, std::vector<float> &edgeCosts, int subsetsSize){
  //calculate the lower bounds q(e)
  std::vector<bool> vVisited(subsetsSize, false);
  std::vector<bool> eVisited(edgeList.size(), false);
  std::vector<float> r(subsetsSize, 0);
  std::vector<float> q(edgeList.size(), 0);
  unsigned int unvisitedV = subsetsSize;
  while(unvisitedV>0){//while there exist an unvisited vertex
    //choose unvisited vertex with largest r
    float maxR=-10;
    int maxV=-1;
    for(unsigned int j=0; j<r.size(); j++){
      if(vVisited[j]==false && r[j]>maxR){
        maxR=r[j];
        maxV=j;
      }
    }
    //for each unvisited adjacent node to maxV
    //please note: here a lookup would help to reduce the calls (check vertices instead of edges)
    for(unsigned j=0; j<edgeList.size(); j++){
      if(eVisited[j]==false){//unvisited
        int neighbour = -1;
        if(edgeList[j].x != maxV && edgeList[j].y != maxV){//not adjacent
        }else if(edgeList[j].x == maxV){
          neighbour=edgeList[j].y;
        }else{
          neighbour=edgeList[j].x;
        }
        if(neighbour>=0){//adjacent
          r[neighbour]+=edgeCosts[j];
          q[j]=r[neighbour];
          eVisited[j]=true;
        }
      }
    }
    vVisited[maxV] = true;
    unvisitedV--;
  }
  return q;
}


float GraphCut::initialLambda(Mat &adjacencyMatrix, int &lambda_id){
  float lambda_score = 1000000;
  for(unsigned int j=0; j<adjacencyMatrix.size().height; j++){
    float score=0;
    for(unsigned int k=0; k<adjacencyMatrix.size().width; k++){
      score+=adjacencyMatrix.at<float>(j,k);
    }
    if(score<lambda_score){
      lambda_score=score;
      lambda_id=j;
    }
  }
  return lambda_score;
}


void GraphCut::createEdgeList(Mat &adjacencyMatrix, std::vector<Point> &edgeList, std::vector<float> &edgeCosts){
  for(unsigned int j=0; j<adjacencyMatrix.size().height; j++){
    for(unsigned int k=j+1; k<adjacencyMatrix.size().width; k++){
      if(adjacencyMatrix.at<float>(j,k)>0){
        Point p(j,k);
        edgeList.push_back(p);
        edgeCosts.push_back(adjacencyMatrix.at<float>(j,k));
      }
    }
  }
}


std::vector<std::vector<int> > GraphCut::createInitialNodes(Mat &adjacencyMatrix){
  std::vector<std::vector<int> > subsets;
  for(unsigned int j=0; j<adjacencyMatrix.size().height; j++){
    std::vector<int> sg;
    sg.push_back(j);
    subsets.push_back(sg);
  }
  return subsets;
}


float GraphCut::merge(std::vector<Point> &edgeList, std::vector<float> &edgeCosts, std::vector<float> &q,
                                  std::vector<std::vector<int> > &subsets, float lambda_score, int j, int &lambda_id){
  int x=edgeList[j].x;
  int y=edgeList[j].y;
  edgeList.erase(edgeList.begin()+j);//delete the merging edge
  edgeCosts.erase(edgeCosts.begin()+j);
  q.erase(q.begin()+j);
  std::vector<std::vector<int> > sameIds(subsets.size());//merge edges
  for(unsigned int k=0; k<edgeList.size(); k++){
    if(edgeList[k].y == y){//relabel edges with y to x
      edgeList[k].y=x;
      if(edgeList[k].y<edgeList[k].x){//switch (first always smaller)
        edgeList[k].y=edgeList[k].x;
        edgeList[k].x=x;
      }
    }
    if(edgeList[k].x == y){
      edgeList[k].x=x;
      if(edgeList[k].y<edgeList[k].x){
        edgeList[k].x=edgeList[k].y;
        edgeList[k].y=x;
      }
    }
    if(edgeList[k].x==x){
      sameIds[edgeList[k].y].push_back(k);//save double edges for merge
    }
    else if(edgeList[k].y==x){
      sameIds[edgeList[k].x].push_back(k);
    }
  }
  //merge edges
  for(unsigned int k=0; k<sameIds.size(); k++){
    if(sameIds[k].size()>1){// two edges connecting the same pair of vertices
      float maxQ = q[sameIds[k][0]];//update q with highest cost of merged edges
      for(unsigned int l=1; l<sameIds[k].size(); l++){
        edgeCosts[sameIds[k][0]]+=edgeCosts[sameIds[k][l]];//sum edge costs
        if(q[sameIds[k][l]]>maxQ) maxQ=q[sameIds[k][l]]; //update q
      }
      q[sameIds[k][0]]=maxQ;//new q is the max of all merged q??s
    }
  }

  //delete merged edges (edges, costs, and q??s)
  std::vector<int> merged;
  for(unsigned int k=0; k<sameIds.size(); k++){
    if(sameIds[k].size()>1){
      for(unsigned int l=1; l<sameIds[k].size(); l++){
        int countSmaller=0;//size for reIndex (multiple edge merges)
        for(unsigned m=0; m<merged.size(); m++){
          if(merged[m]<sameIds[k][l]) countSmaller++;//reIndex
        }
        edgeCosts.erase(edgeCosts.begin()+sameIds[k][l]-countSmaller);
        edgeList.erase(edgeList.begin()+sameIds[k][l]-countSmaller);
        q.erase(q.begin()+sameIds[k][l]-countSmaller);
        merged.push_back(sameIds[k][l]);//without reIndex
      }
    }
  }

  //merge vertices
  for(unsigned int k=0; k<subsets[y].size(); k++){
    subsets[x].push_back(subsets[y][k]);
  }
  subsets.erase(subsets.begin()+y);//delete merged vertex

  //calculate cut cost of new single vertex and update initial score if smaller
  float cx=0;
  for(unsigned int k=0; k<edgeList.size(); k++){
    if(edgeList[k].x == x || edgeList[k].y==x){
      cx+=edgeCosts[k];
    }
    //decrease all ids bigger y (removed vertex)
    if(edgeList[k].x>y) edgeList[k].x--;
    if(edgeList[k].y>y) edgeList[k].y--;

  }
  if(cx<lambda_score) {//update lambda (merged node is new candidate)
    lambda_score=cx;
    lambda_id=x;
  }else if(lambda_id==y){//update lambda-id (old id merged to new one)
    lambda_id=x;
  }else if(lambda_id>y){//decrease lambda-id (due to vertex relabeling)
    lambda_id--;
  }
  return lambda_score;
}


std::vector<std::vector<int> > GraphCut::findUnconnectedSubgraphs(Mat &adjacencyMatrix){
  std::vector<std::vector<int> > subgraphs;
  std::vector<bool> visited(adjacencyMatrix.size().height,false);
  unsigned int visitCount=0;
  while(visitCount<adjacencyMatrix.size().height){//while unassigned vertices
    std::vector<int> subgraph;//create new subgraph
    int next=0;
    for(unsigned int i=0; i<visited.size(); i++){//find first unassigned
      if(visited[i]==false){
        next=i;
        break;
      }
    }
    subgraph.push_back(next);//add first unassigned
    visited[next]=true;
    visitCount++;

    for(unsigned int i=0; i<subgraph.size(); i++){//breadth first search
      for(unsigned int j=0; j<adjacencyMatrix.size().width; j++){
        if(adjacencyMatrix.at<float>(subgraph[i],j)>0 && visited[j]==false){//add node if not assigned and edge exists
          visited[j]=true;
          visitCount++;
          subgraph.push_back(j);
        }
      }
    }
    subgraphs.push_back(subgraph);//add subgraph
  }
  return subgraphs;
}


Mat GraphCut::createSubMatrix(Mat &adjacencyMatrix, std::vector<int> &subgraph){
  Mat subMatrix(static_cast<int>(subgraph.size()),static_cast<int>(subgraph.size()), CV_32FC1);
  for(unsigned int j=0; j<subMatrix.size().height; j++){
    for(unsigned int k=0; k<subMatrix.size().width; k++){
      subMatrix.at<float>(j,k)=adjacencyMatrix.at<float>(subgraph[j],subgraph[k]);
    }
  }
  return subMatrix;
}


Mat GraphCut::calculateProbabilityMatrix(Mat &initialMatrix, bool symmetry){
  Mat probabilities=Mat(initialMatrix.size().height,initialMatrix.size().width,0.0);
  for(unsigned int a=0; a<initialMatrix.size().width; a++){
    int count = 0;
    for(unsigned int b=0; b<initialMatrix.size().width; b++){//count number of edges for each node
      if(initialMatrix.at<bool>(a,b)==1 && initialMatrix.at<bool>(b,a)==1 && a!=b){
        count++;
      }
    }
    for(unsigned int b=0; b<initialMatrix.size().width; b++){//cost of each edge: 1. / num edges
      if(initialMatrix.at<bool>(a,b)==1 && initialMatrix.at<bool>(b,a)==1 && a!=b){
        probabilities.at<float>(b,a)=1./(float)count;
      }
    }
  }
  if(symmetry==true){
    for(unsigned int i=0; i<probabilities.size().width; i++){//symmetry
      for(unsigned int j=i+1; j<probabilities.size().width; j++){
        float v = (probabilities.at<float>(i,j)+probabilities.at<float>(j,i))/2.;
        probabilities.at<float>(i,j)=v;
        probabilities.at<float>(j,i)=v;
      }
    }
  }
  return probabilities;
}


void GraphCut::mergeMatrix(Mat &dst, Mat &src){
  if(src.size().height!=dst.size().height){
    throw ("unequal sizes");
  }
  for(unsigned int i=0; i<src.size().height; i++){
    for(unsigned int j=0; j<src.size().width; j++){
      if(src.at<bool>(i,j)==true) dst.at<bool>(i,j)=true;
    }
  }
}


void GraphCut::weightMatrix(Mat &dst, Mat &featureMatrix, float weight){
  if(featureMatrix.size().height!=dst.size().height){
    throw ("unequal sizes");
  }
  for(unsigned int i=0; i<featureMatrix.size().height; i++){
    for(unsigned int j=0; j<featureMatrix.size().width; j++){
      if(featureMatrix.at<bool>(i,j)==true) dst.at<float>(i,j)*=weight;
    }
  }
}
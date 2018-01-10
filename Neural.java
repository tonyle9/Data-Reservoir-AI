
package s6regen;

// Evolve deep random projection neural network class

class Neural {

  final int vecLen;
  final int density;
  final int depth;
  final int precision;
  final long hash;
  float parentCost;
  float[][][] weights;
  float[][][] mWeights;
  float[] workA;
  float[] workB;
  float[] workC;
  float[] workD;
  RNG rnd;

  // vecLen must be 2,4,8,16,32.....
  Neural(int vecLen, int density, int depth,int precision) {
    this.vecLen=vecLen;
    this.density=density;
    this.depth=depth;
    this.precision=precision;
    parentCost=Float.POSITIVE_INFINITY;
    weights=new float[depth][density][vecLen];
    mWeights=new float[depth][density][vecLen];
    workA=new float[vecLen];
    workB=new float[vecLen];
    workC=new float[vecLen];
    workD=new float[vecLen];   
    rnd=new RNG();
    hash=rnd.nextLong();
    for (int i=0; i<depth; i++) {
      for (int j=0; j<density; j++) {
        for (int k=0; k<vecLen; k++) {
          weights[i][j][k]=rnd.nextFloatSym();
        }
      }
    } 
    float sc=1f/(float)Math.sqrt(density);
    for (int j=0; j<density; j++) {
      VecOps.scale(weights[depth-1][j], weights[depth-1][j], sc);
    }
  }

  void recall(float[] resultVec, float[] inVec) {
    VecOps.adjust(workA, inVec);
    long h=hash;
    for (int i=0; true; i++) {
      System.arraycopy(workA, 0, workB, 0, vecLen);
      java.util.Arrays.fill(resultVec, 0f);
      for (int j=0; j<density; j++) {
        WHT.fastRP(workA, h++);
        WHT.fastRP(workB, h++);
        VecOps.multiply(workC, workA, workB);
        WHT.fastRP(workC, h++);
        VecOps.multiplyAddTo(resultVec, workC, weights[i][j]);
      }
      if (i==depth-1) break;
      VecOps.adjust(workA, resultVec);
    }
  }
  
// the elements of targetVecs should have a magnitude of around 0 to 1.
  void train(float[][] targetVecs, float[][] inputVecs) {
    for (int i=0; i<depth; i++) {  // create mutated weights
      for (int j=0; j<density; j++) {
        for (int k=0; k<vecLen; k++) {
          mWeights[i][j][k]=rnd.mutateXSym(weights[i][j][k], precision);
        }
      }
    }
    float[][][] t=weights;  // swap the weights with the mutated weights
    weights=mWeights;
    mWeights=t;
    float cCost=0f;  //evaluate the cost of the mutated weights.
    for (int i=0; i<targetVecs.length; i++) {  
      recall(workD, inputVecs[i]);
      VecOps.subtract(workD, targetVecs[i], workD);
      cCost+=VecOps.sumSq(workD);
    }
    if (cCost<=parentCost) {
      parentCost=cCost;  // keep the mutated weights as new parent
    } else {  // else swap back the old unmutated parent weights
      t=weights;
      weights=mWeights;
      mWeights=t;
    }
  }
  
  float getCost(){
    return parentCost;
  }
}
/* Associative memory class
With locality sensitive hashing (LSH) only a few bits change with a small change in input.
Here in each dimension some output bits of a LSH (-1,1 bipolar) are weighted and summed.
For a particular input the weights are easily (and equally) adjusted to give a required output.
For a previously stored value retrived again the newly stored adjustment will tend to cancel
out to zero (with a Gaussian noise residue), assuming the LHS bit response to the two inputs
is quite different bit-wise. 
 */
package s6regen;

import java.io.Serializable;
import java.util.Arrays;

class AM implements Serializable {

    int vecLen;
    int density;
    long hash;
    float[][] weights;
    float[][] bipolar;
    float[] workA;
    float[] workB;

    // vecLen must be 2,4,8,16,32.....
    // density is the maximum number of vector pairs that can be associated with
    // repeated training.
    AM(int vecLen, int density) {
        this.vecLen = vecLen;
        this.density = density;
        hash = System.nanoTime();
        weights = new float[density][vecLen];
        bipolar = new float[density][vecLen];
        workA = new float[vecLen];
        workB = new float[vecLen];
    }

    public void recallVec(float[] resultVec, float[] inVec) {
        System.arraycopy(inVec, 0, workA, 0, vecLen);
        Arrays.fill(resultVec, 0f);
        for (int i = 0; i < density; i++) {
            WHT.fastRP(workA, hash + i);
            WHT.signOf(bipolar[i], workA);
            VecOps.multiplyAddTo(resultVec, weights[i], bipolar[i]);
        }
    }

    public void trainVec(float[] targetVec, float[] inVec) {
        float rate = 1f / density;
        recallVec(workB, inVec);
        for (int i = 0; i < vecLen; i++) {
            workB[i] = (targetVec[i] - workB[i]) * rate;    //get the error term in workB
        }
        for (int i = 0; i < density; i++) {                       // correct the weights 
            VecOps.multiplyAddTo(weights[i], workB, bipolar[i]);  // to give the required output
        }
    }
    
    public void reset(){
        for(float[] x:weights){
            Arrays.fill(x, 0f);
        }
    }
}

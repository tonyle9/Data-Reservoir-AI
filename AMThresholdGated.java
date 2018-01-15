// Associative memory class, threshold gated write to allow long term memory.
package s6regen;

import java.io.Serializable;
import java.util.Arrays;

public final class AMThresholdGated implements Serializable {

    private final int vecLen;
    private final int density;
    private final float threshold;
    private final long hash;
    private final float[][] weights;
    private final float[][] bipolar;
    private final float[] workA;
    private final float[] workB;

    // vecLen must be 2,4,8,16,32.....
    // density is the maximum number of vector pairs that can be associated with
    // repeated training.
    public AMThresholdGated(int vecLen, int density,float threshold) {
        this.vecLen = vecLen;
        this.density = density;
        this.threshold=threshold;
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

    // If target after truncation equals zero no value is stored, otherwise
    // the truncated value is stored.  Giving (self gated) storage.
    public void trainVec(float[] targetVec, float[] inVec) {
        float rate = 1f / density;
        recallVec(workB, inVec);
        VecOps.truncate(workA,targetVec, threshold);    // truncate the target
        for (int i = 0; i < vecLen; i++) {
            workB[i] = (workA[i] - workB[i]) * rate;    //get the error term in workB
        }
        for (int i = 0; i < density; i++) {             // correct the weights 
            float[] wt=weights[i],bi=bipolar[i];
            for(int j=0;j<vecLen;j++){
                if(workA[j]!=0f){   // if not gated out by truncation update the weight
                    wt[j]+=workB[j]*bi[j];
                }
            }
        }
    }

    public void reset() {
        for (float[] x : weights) {
            Arrays.fill(x, 0f);
        }
    }
}

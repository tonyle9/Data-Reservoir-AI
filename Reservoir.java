// Contains most of the information for the Data Reservoir Compute AI.
// Including all trainable weights for all Compute class operators.
// One reason for this is to simplify access for evolution based algorithms.
// All Compute based subclasses operate on computeSize-ed data arrays.
// They get data using the gather method which uses random projection 
// base dimension reduction of the entire reservoir.  A change in one single
// value in the reservoir produces a unique pattern change in the gathered data.
// The gather method also can select specific places in the reservoir to get
// information from because of weighting prior to the random projection process.
// The reservoir is composed of 3 parts <input><writable section><general>
// When a compute object is finished it can either scatter the result to a
// specific place in the writable section or again use random projection with
// weighting based selection to write to the general section.
// This should allow for complex connectivity (eg. modualar) to emerge.
package s6regen;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;

public class Reservoir implements Serializable {

    final static float MIN_SQ = 1e-20f;
    final int computeSize;
    private final int reservoirSize;
    private final int inputSize;
    private final int writableSize;
    private final int outputSize;

    int hashIndex;
    int weightIndex;

    private int weightSize;
    float[] weights;
    private transient float[] reservoir;
    transient float[][] computeBuffers;
    private transient RNG rng;
    private final ArrayList<Compute> list; // list of all compute units for the AI

    public Reservoir(int computeSize, int reservoirSize, int inputSize, int writableSize, int outputSize) {
        assert computeSize >= 16 : "Requirement for WTH class";
        assert (computeSize & (computeSize - 1)) == 0 : "Power of 2 for WHT class";
        assert reservoirSize % computeSize == 0 : "Make multiple of computeSize";
        assert inputSize % computeSize == 0 : "Make multiple of computeSize";
        assert writableSize % computeSize == 0 : "Make multiple of computeSize";

        this.computeSize = computeSize;
        this.reservoirSize = reservoirSize;
        this.inputSize = inputSize;
        this.writableSize = writableSize;
        this.outputSize = outputSize;
        list = new ArrayList<>();
    }

    public void addComputeUnit(Compute c) {
        list.add(c);
    }

// Call after adding compute units. Don't add more compute units and call again.
// Also sets up after deserialization. Called from the readObject() method.    
    public void prepareForUse() {
        reservoir = new float[reservoirSize];  //transient
        int bN = 0;
        for (Compute c : list) {
            weightSize += c.weightSize();
            if (bN < c.buffersRequired()) {
                bN = c.buffersRequired();
            }
        }
        computeBuffers = new float[bN][computeSize];    //transient
        rng = new RNG();                                //transient
        if (weights == null) {  // if deserializing weights typically != null
            weights = new float[weightSize];
            for (int i = 0; i < weightSize; i++) {  // randomize inital weights.
                weights[i] = rng.nextFloatSym();
            }
        }
    }

    public void computeAll() {
        hashIndex = 0;
        weightIndex = 0;
        for (Compute c : list) {
            c.compute();
        }
        assert weightIndex == weightSize : "Error in Compute subclass weightIndex or weightSize";
    }

// clears all held state such as in associative memory.    
    public void resetHeldStateAll() {
        for (Compute c : list) {
            c.resetHeldState();
        }
    }

    public void setInput(float[] input) {
        System.arraycopy(input, 0, reservoir, 0, inputSize);
    }

    public void getOutput(float[] output) {
        System.arraycopy(reservoir, inputSize + writableSize, output, 0, outputSize);
    }

    public void mutate(long mutatePrecision) {
        for (int i = 0; i < weightSize; i++) {
            weights[i] = rng.mutateXSym(weights[i], mutatePrecision);
        }
    }

    public int getWeightSize() {
        return weightSize;
    }

    public void getWeights(float[] vec) {
        System.arraycopy(weights, 0, vec, 0, weightSize);
    }

    public void setWeights(float[] vec) {
        System.arraycopy(vec, 0, weights, 0, weightSize);
    }

    void gather(float[] g) {
        int wtIdx = weightIndex; // Get as local as an optimization
        float[] wt = weights;    // Check later if this is so necessary
        float[] res = reservoir;
        int i = 0;
        while (i < computeSize) {
            g[i] = res[i] * wt[wtIdx++];
            i++;
        }
        while (i < reservoirSize) {
            WHT.fastRP(g, hashIndex++);
            for (int j = 0; j < computeSize; j++) {
                g[j] += res[i++] * wt[wtIdx++];
            }
        }
        weightIndex = wtIdx;  // set weightIndex to correct value
        WHT.fastRP(g, hashIndex++);
    }

// s is destroyed by this method, it is assumed the compute unit will not need
// it again.
    void scatter(float[] s) {
        int wtIdx = weightIndex; // Get as local as an optimization
        float[] wt = weights;
        float[] res = reservoir;
        int i = inputSize + writableSize;
        while (i < reservoirSize) {
            WHT.fastRP(s, hashIndex++);
            /*    for (int j = 0; j < computeSize; j++) {
                float p=wt[wtIdx++];
                if(p>=0f){  //  if p<0 then leave the reservoir value unchanged.
                    res[i]= s[j]*p+(1f-p)*res[i]; // otherwise blend the two.
                }
                i++; 
            } */
            for (int j = 0; j < computeSize; j++) {
                float p = wt[wtIdx++];
                p*=p; // smooth non-linear blending to make things easier for evolution, if wt is low eg 0.1
                res[i] = s[j] * p + (1f - p) * res[i]; // the reservoir is hardly effected
                i++;
            }
        }
        weightIndex = wtIdx;  // put back the new index
    }

    void scatterWritable(float[] s, int location) {
        System.arraycopy(s, 0, reservoir, inputSize + location, computeSize);
    }

    int sizeGather() {
        return reservoirSize;
    }

    int sizeScatter() {
        return reservoirSize - inputSize - writableSize;
    }

    void normalizeGeneral() {
        int start = inputSize + writableSize;
        float sumSq = 0f;
        for (int i = start; i < reservoirSize; i++) {
            sumSq += reservoir[i] * reservoir[i];
        }
        float adj = 1f / (float) Math.sqrt((sumSq / (reservoirSize - start)) + MIN_SQ);
        for (int i = start; i < reservoirSize; i++) {
            reservoir[i] *= adj;
        }
    }

    void normalizeInput() {
        float sumSq = 0f;
        for (int i = 0; i < inputSize; i++) {
            sumSq += reservoir[i] * reservoir[i];
        }
        float adj = 1f / (float) Math.sqrt((sumSq / inputSize) + MIN_SQ);
        for (int i = 0; i < inputSize; i++) {
            reservoir[i] *= adj;
        }
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        prepareForUse(); //Set up all the buffers and working arrays
    }
}

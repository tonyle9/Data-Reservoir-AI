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
// This should allow for complex connectivity (eg. modualar) to emerge.  As well
// as recurrence.
package s6regen;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;

public class Reservoir implements Serializable {

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

    Reservoir(int computeSize, int reservoirSize, int inputSize, int writableSize, int outputSize) {
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

    Reservoir(Reservoir r) {
        computeSize = r.computeSize;
        reservoirSize = r.reservoirSize;
        inputSize = r.inputSize;
        writableSize = r.writableSize;
        outputSize = r.outputSize;
        list = new ArrayList<>(r.list);
    }

    public void addComputeUnit(Compute c) {
        list.add(c);
    }
    
// Call after adding compute units
    public void prepareForUse() {
        reservoir = new float[reservoirSize];
        int bN = 0;
        for (Compute c : list) {
            weightSize += c.weightSize();
            if (bN < c.buffersRequired()) {
                bN = c.buffersRequired();
            }
        }
        computeBuffers = new float[bN][computeSize];
        rng = new RNG();
        if (weights == null) {
            weights = new float[weightSize];
            for(int i=0;i<weightSize;i++){
                weights[i]=rng.nextFloatSym();
            }
        }
    }

    public void computeAll() {
        hashIndex = 0;
        weightIndex = 0;
        for (Compute c : list) {
            c.compute();
        }
    }

    public void setInput(float[] input) {
        System.arraycopy(input, 0, reservoir, 0, inputSize);
    }

    public void getOutput(float[] output) {
        System.arraycopy(reservoir, inputSize + writableSize, output, 0, outputSize);
    }

    public void copyWeightsMutateFrom(Reservoir r, long mutatePrecision) {
        for (int i = 0; i < weightSize; i++) {
            weights[i] = rng.mutateXSym(r.weights[i], mutatePrecision);
        }
    }

    void gather(float[] g) {
        int mask = computeSize - 1;
        for (int i = 0; i < computeSize; i++) {
            g[i] = reservoir[i] * weights[weightIndex++];
        }
        for (int i = computeSize; i < reservoirSize; i++) {
            if ((i & mask) == 0) {
                WHT.fastRP(g, hashIndex++);
            }
            g[i & mask] = +reservoir[i] * weights[weightIndex++];
        }
        WHT.fastRP(g, hashIndex++);
    }

    void scatter(float[] s) {
        int mask = computeSize - 1;
        for (int i = inputSize + writableSize; i < reservoirSize; i++) {
            if ((i & mask) == 0) {
                WHT.fastRP(s, hashIndex++);
            }
            reservoir[i] += s[i & mask] * weights[weightIndex++];
        }
    }

    void scatterWritable(float[] s, int location) {
        System.arraycopy(s, 0, reservoir, inputSize + location, s.length);
    }
    
    int sizeGather(){
        return reservoirSize;
    }
    
    int sizeScatter(){
        return reservoirSize-inputSize-writableSize;
    }
    
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        prepareForUse(); //Set up all the buffers and working arrays
    }
}

// Abstract class that takes a Reservoir object as a sort of parent.
// The parent provides a number of resources to Compute object such
// as the weights array and weight index for subclasses to use.
// The subclasses select some part of the reservoir to operate on
// using the reservoir gather method, do a computation and then
// send the result back to some part of the reservoir using 
// the scatter method.  The selected places being decided by
// information in the weights array (or a specific place when
// using scatterWritable.
package s6regen;
public abstract class Compute implements java.io.Serializable {
    
    final Reservoir reservoir;
    
    public Compute(Reservoir r){
        reservoir=r;
    }
    
    public void resetHeldState(){
    }
    
    public abstract void compute();

    public abstract int weightSize();
    
    public abstract int buffersRequired();
    
}

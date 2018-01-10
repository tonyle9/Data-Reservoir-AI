package s6regen;
public abstract class Compute implements java.io.Serializable {
    
    final Reservoir reservoir;
    
    public Compute(Reservoir r){
        reservoir=r;
    }
    
    public abstract void compute();

    public abstract int weightSize();
    
    public abstract int buffersRequired();
    
}

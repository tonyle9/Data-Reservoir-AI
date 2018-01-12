package s6regen;

public abstract class Evolve{

    private final Reservoir reservoir;
    private long precision;
    private final float[] parent;
    private float parentCost;
    
//  Use fully perpared reservoir. Higher precision equals lower mutation rate.
    public Evolve(Reservoir r,long precision){
        reservoir=r;
        this.precision=precision;
        parent=new float[r.getWeightSize()];
        parentCost=Float.POSITIVE_INFINITY;
    }
    
//  Next evolution evaluation.
    public void iterate(){
        reservoir.getWeights(parent);
        reservoir.mutate(precision);    // Reservoir now contains child weights
        float childCost=evaluateCost(reservoir);
        if(childCost<parentCost){
            parentCost=childCost;
        } else {
            reservoir.setWeights(parent);
        }
    }
    
    public void setPrecision(long pre){
        precision=pre;
    }
    
    public long getPrecision(){
        return precision;
    }
    
    public abstract float evaluateCost(Reservoir r);
       
}

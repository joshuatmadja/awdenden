import java.io.Serializable;

public class Weight implements Serializable {
    private double value;
    private int node_in;
    private int node_out;
    
    public Weight() {
        //System.out.println("Weight CTOR");
    }
    
    //SETTER
    public void setValue(final double value) {
        this.value = value;
    }
    
    public void setNodeIn(final int node_in) {
        this.node_in = node_in;
    }
    
    public void setNodeOut(final int node_out) {
        this.node_out = node_out;
    }
    
    //GETTER
    public double getValue() {
        return this.value;
    }
    
    public int getNodeIn() {
        return this.node_in;
    }
    
    public int getNodeOut() {
        return this.node_out;
    }
}

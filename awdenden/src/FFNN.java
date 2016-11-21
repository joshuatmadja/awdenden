import java.util.Vector;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class FFNN extends AbstractClassifier {
    private double learning_rate;
    private int n_in;
    private int n_out;
    private int n_hidden;
    
    public FFNN()  {
        System.out.println("FFNN CTOR");
    }
    
    //SETTER
    public void setLearningRate(final double learning_rate) {
        this.learning_rate = learning_rate;
    }
    
    public void setNIn(final int n_in) {
        this.n_in = n_in;
    }
    
    public void setNOut(final int n_out) {
        this.n_out = n_out;
    }
    
    public void setNHidden(final int n_hidden) {
        this.n_hidden = n_hidden;
    }
    
    //GETTER
    public double getLearningRate() {
        return this.learning_rate;
    }
    
    public int getNIn() {
        return this.n_in;
    }
    
    public int getNOut() {
        return this.n_out;
    }
    
    public int getNHidden() {
        return this.n_hidden;
    }
    
    //
    @Override
    public void buildClassifier(Instances i) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}

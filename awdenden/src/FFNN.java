import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class FFNN extends AbstractClassifier {
    private double learning_rate;
    private int n_in;
    private int n_out;
    private int n_hidden;
    private final ArrayList<Weight> weights;
    private final ArrayList<Double> nodes;
    private final ArrayList<Double> errors;
    
    public FFNN()  {
        weights = new ArrayList<>();
        nodes = new ArrayList<>();
        errors = new ArrayList<>();
        //System.out.println("FFNN CTOR");
    }
    
    //SETTER
    public void setLearningRate(final double learning_rate) {
        this.learning_rate = learning_rate;
    }
    
    public void setNIn(final int n_in) {
        this.n_in = n_in+1;
    }
    
    public void setNOut(final int n_out) {
        this.n_out = n_out;
    }
    
    public void setNHidden(final int n_hidden) {
        if (n_hidden > 0) {
            this.n_hidden = n_hidden+1;
        } else {
            this.n_hidden = n_hidden;
        }
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
    
    public ArrayList<Weight> getWeights() {
        return weights;
    }
    
    public ArrayList<Double> getNodes() {
        return nodes;
    }
    
    public ArrayList<Double> getErrors() {
        return errors;
    }
    
    public int getNTotal() {
        return (this.n_in + this.n_out + this.n_hidden);
    }
    
    public int getWTotal() {
        if (this.n_hidden == 0) {
            return (this.n_in * this.n_out);
        } else {
            return ((this.n_in * (this.n_hidden - 1)) + (this.n_hidden * this.n_out));
        }
    }
    
    //
    public double nettValue(int nodeOut) {
        //System.out.println("hai " + nodeOut);
        double result = 0.0;
        for (int i = 0; i < this.getWTotal(); ++i) {
            if (this.weights.get(i).getNodeOut() == nodeOut) {
                /*System.out.println("---");
                System.out.println(this.weights.get(i).getValue());
                System.out.println("+++");
                */
                result += this.nodes.get(this.weights.get(i).getNodeIn()) * this.weights.get(i).getValue();
                //System.out.println(nodeOut);
            }
        }
        //System.out.println(result);
        return result;
    }
    
    public double outputValue(double nett) {
        return (1.0/(1.0+(Math.exp((-1.0)*nett))));
    }
    
    public double errorOutput(double output, double target) {
        return (output*(1-output)*(target-output));
    }
    
    public double errorHidden(int nodeIn, double output) {
        double result = 0.0;
        for (int i = 0; i < this.getWTotal(); ++i) {
            if (this.weights.get(i).getNodeIn() == nodeIn) {
                result += this.errors.get(this.weights.get(i).getNodeOut()) * this.weights.get(i).getValue();
            }
        }
        return (output*(1-output)*result);
    }
    
    public double getNewWeight(int i) {
        return (this.weights.get(i).getValue()
                + this.learning_rate 
                * this.nodes.get(this.weights.get(i).getNodeIn()) 
                * this.errors.get(this.weights.get(i).getNodeOut()));
    }
    
    @Override
    public void buildClassifier(Instances dataset) throws Exception {
        /*Mengosongkan weights, nodes, dan errors*/
        this.weights.clear();
        this.nodes.clear();
        this.errors.clear();
        
        /*Inisialisasi weights*/
        int counter = 0;
        while (counter < this.getWTotal()) {
            for (int j = 0; j < (this.getNTotal() - this.n_out); ++j) { //index node_in
                int batas_awal;
                int batas_akhir;
                if ((j < this.n_in) && (this.n_hidden != 0)) {
                    batas_awal = this.n_in + 1;
                    batas_akhir = this.n_in + this.n_hidden;
                } else {
                    batas_awal = this.n_in + this.n_hidden;
                    batas_akhir = this.getNTotal();
                }
                for (int k = batas_awal; k < batas_akhir; ++k) { //index node_out
                    this.weights.add(new Weight());
                    this.weights.get(counter).setValue(new Random().nextDouble() * 0.1 - 0.05);
                    this.weights.get(counter).setNodeIn(j);
                    this.weights.get(counter).setNodeOut(k);
                    ++counter;
                }
            }
        }
        
        /*Inisialisasi nodes dan errors*/
        for (int i = 0; i < this.getNTotal(); ++i) {
            this.nodes.add(1.0);
            this.errors.add(0.0);
        }
        
        int loop = 0;
        while (loop < 1) {
            for (int i = 0; i < dataset.numInstances(); ++i) { //index dataset
                /*Inisialisasi node input*/
                for (int j = 1; j < this.n_in; ++j) { //index input
                    this.nodes.set(j, dataset.instance(i).value(j-1));
                }
                
                //-->Mendapatkan nilai output
                for (int j = this.n_in; j < this.getNTotal(); ++j) { //index output
                    if (!((j == this.n_in) && (this.n_hidden != 0))) {
                        this.nodes.set(j, this.outputValue(this.nettValue(j)));
                    }
                    //System.out.println(nodes.get(j));
                }
                
                //<--Mendapatkan nilai error
                double target = dataset.instance(i).classValue();
                for (int j = this.getNTotal()-1; j >= this.n_in; --j) {
                    if ((j == this.n_in) && (this.n_hidden != 0)) {
                        //Do Nothing
                    } else if (j >= this.getNTotal() - this.n_out) { //errorOutput
                        double output = j - this.getNTotal() + this.n_out;
                        if (output == target) {
                            this.errors.set(j, this.errorOutput(j, 1.0));
                        } else {
                            this.errors.set(j, this.errorOutput(j, 0.0));
                        }
                    } else { //errorHidden
                        //System.out.println(j);
                        this.errors.set(j, this.errorHidden(j, this.nodes.get(j)));
                    }
                    //System.out.println(this.errors.get(j));
                }
                //System.out.println(this.errors.get(1));
                
                //Set nilai baru pada weights
                for (int j = 0; j < this.getWTotal(); ++j) {
                    this.weights.get(j).setValue(this.getNewWeight(j));
                }
                //System.out.println(weights.get(34).getValue());
                /*
                System.out.println("begin");
                System.out.println(nodes.get(10));
                System.out.println(nodes.get(11));
                System.out.println(nodes.get(12));
                System.out.println("end");
                */
            }
            ++loop;
        }
        this.nodes.clear();
        this.errors.clear();
    }
    
    @Override
    public double classifyInstance(Instance data) throws Exception {
        /*Mengosongkan nodes*/
        this.nodes.clear();
        
        /*Inisialisasi nodes*/
        for (int i = 0; i < this.getNTotal(); ++i) {
            this.nodes.add(1.0);
        }
        
        //-->
        for (int j = this.n_in; j < this.getNTotal(); ++j) { //index output
            if (!((j == this.n_in) && (this.n_hidden != 0))) {
                this.nodes.set(j, this.outputValue(this.nettValue(j)));
            }
        }
        
        double result = 0.0;
        if (data.numClasses() > 2) {
            double counter = 0.0;
            double result_class = 0.0;
            double result_output = nodes.get(this.n_in + this.n_hidden);
            for (int k = this.n_in + this.n_hidden + 1; k < this.getNTotal(); ++k) {
                ++counter;
                if (result_output < nodes.get(k)) {
                    result_output = nodes.get(k);
                    result = counter;
                }
            }
        } else {
            if (nodes.get(nodes.size()-1) > 0.5) {
                result = 0.0;
            } else {
                result = 1.0;
            }
        }
        
        this.nodes.clear();
        return result;
    }
}

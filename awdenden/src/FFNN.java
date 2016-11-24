import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
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
        double result = 0.0;
        for (int i = 0; i < this.getWTotal(); ++i) {
            if (this.weights.get(i).getNodeOut() == nodeOut) {
                result += (this.nodes.get(this.weights.get(i).getNodeIn()) * this.weights.get(i).getValue());
            }
        }
        return result;
    }
    
    public double outputValue(double nett) {
        return (1.0/(1.0+(Math.exp((-1.0)*nett))));
    }
    
    public double errorOutput(double output, double target) {
        return (output * (1.0 - output) * (target - output));
    }
    
    public double errorHidden(int nodeIn) {
        double sigma = 0.0;
        double output = this.nodes.get(nodeIn);
        for (int i = 0; i < this.getWTotal(); ++i) {
            if (this.weights.get(i).getNodeIn() == nodeIn) {
                sigma += (this.weights.get(i).getValue() * this.errors.get(this.weights.get(i).getNodeOut()));
            }
        }
        return (output * (1.0 - output) * sigma);
    }
    
    public double getNewWeight(int i) {
        return (this.weights.get(i).getValue() + (this.learning_rate * this.nodes.get(this.weights.get(i).getNodeIn()) * this.errors.get(this.weights.get(i).getNodeOut())));
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
        
        ArrayList<Double> temp1 = new ArrayList<>();
        ArrayList<Double> temp2 = new ArrayList<>();
        int loop = 0;
        boolean isStop = false;
        while (!isStop) {
            for (int i = 0; i < dataset.numInstances(); ++i) { //index dataset
                //-->Mengisi node input
                for (int j = 1; j < this.n_in; ++j) { //index input
                    this.nodes.set(j, dataset.instance(i).value(j-1));
                }
                
                //-->Mengisi node hidden dan output
                for (int j = this.n_in; j < this.getNTotal(); ++j) { //index output
                    if (!((j == this.n_in) && (this.n_hidden != 0))) {
                        this.nodes.set(j, this.outputValue(this.nettValue(j)));
                    }
                }
                
                //<--Mengisi nilai error
                double target = dataset.instance(i).classValue();
                for (int j = this.getNTotal() - 1; j >= this.n_in; --j) {
                    if ((j == this.n_in) && (this.n_hidden != 0)) {
                        //Do Nothing
                    } else if (j >= this.getNTotal() - this.n_out) { //errorOutput
                        if (this.n_out == 1) {
                            if (this.nodes.get(j) >= 0.5) {
                                this.errors.set(j, this.errorOutput(this.nodes.get(j), 1.0));
                            } else {
                                this.errors.set(j, this.errorOutput(this.nodes.get(j), 0.0));
                            }
                        } else {
                            double output = j - this.getNTotal() + this.n_out;
                            if (output == target) {
                                this.errors.set(j, this.errorOutput(this.nodes.get(j), 1.0));
                            } else {
                                this.errors.set(j, this.errorOutput(this.nodes.get(j), 0.0));
                            }
                        }
                    } else { //errorHidden
                        this.errors.set(j, this.errorHidden(j));
                    }
                }
                
                //-->Mengisi nilai weights yang baru
                for (int j = 0; j < this.getWTotal(); ++j) {
                    this.weights.get(j).setValue(this.getNewWeight(j));
                }
            }
            if (loop == 0) {
                for (int j = this.getNTotal() - this.n_out; j < this.getNTotal(); ++j) {
                    temp1.add(this.nodes.get(j));
                }
            } else {
                isStop = true;
                for (int j = this.getNTotal() - this.n_out; j < this.getNTotal(); ++j) {
                    temp2.add(this.nodes.get(j));
                }
                for (int j = 0; j < this.n_out; ++j) {
                    if ((temp1.get(j) - temp2.get(j) > 0.000005) || (temp1.get(j) - temp2.get(j) < -0.000005)) {
                        isStop = false;
                    }
                    temp1.set(j, temp2.get(j));
                }
                temp2.clear();
            }
            ++loop;
        }
        this.nodes.clear();
        this.errors.clear();
    }
    
    @Override
    public double classifyInstance(Instance data) throws Exception {
        double result = 0.0;
        double[] results = this.distributionForInstance(data);
        double max = -1e9;
        
        if (this.n_out == 1) {
            if (results[0] >= 0.5) {
                result = 0.0;
            } else {
                result = 1.0;
            }
        } else {
            for (int i = 0; i < this.n_out; ++i) {
                if (results[i] > max) {
                    result = (double) i;
                    max = results[i];
                }
            }
        }
        
        return result;
    }
    
    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        double[] result = new double[this.n_out];
        
        /*Mengosongkan nodes*/
        this.nodes.clear();
        
        /*Inisialisasi nodes*/
        for (int i = 0; i < this.getNTotal(); ++i) {
            this.nodes.add(1.0);
        }
        
        //-->Mengisi node input
        for (int j = 1; j < this.n_in; ++j) { //index input
            this.nodes.set(j, instnc.value(j-1));
        }
                
        //-->Mengisi node hidden dan output
        for (int j = this.n_in; j < this.getNTotal(); ++j) { //index output
            if (!((j == this.n_in) && (this.n_hidden != 0))) {
                this.nodes.set(j, this.outputValue(this.nettValue(j)));
            }
        }
        
        int counter = 0;
        for (int i = this.n_in + this.n_hidden; i < this.getNTotal(); ++i) {
            result[counter] = this.nodes.get(i);
            ++counter;
        }
        
        this.nodes.clear();
        return result;
    }
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities((CapabilitiesHandler) this);
        result.enableAll();
        return result;
    }
}

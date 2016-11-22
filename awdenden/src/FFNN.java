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
    private ArrayList<Weight> weights;
    
    public FFNN()  {
        weights = new ArrayList<>();
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
    
    public int getTotalN() {
        return (this.n_in + this.n_out + this.n_hidden);
    }
    
    //
    public double nettValue(int nodeOut, ArrayList<Double> node) {
        double result = 0.0;
        int i = 0;
        while (weights.get(i).getNodeOut() != nodeOut) {
            ++i;
        }
        
        while (weights.get(i).getNodeOut() == nodeOut) {
            Weight w = weights.get(i);
            result += node.get(w.getNodeIn()) * w.getValue();
            ++i;
        }
        return result;
    }
    
    public double outputValue(double x) {
        //System.out.println(1/(1+(Math.exp(-x))));
        return 1/(1+(Math.exp((-1.0)*x)));
    }
    
    public double errorOutput(double output, double target) {
        return output*(1-output)*(target-output);
    }
    
    public double errorHidden(int nodeIn, double output, ArrayList<Double> error) {
        double result = 0.0;
        int i = 0;
        while (weights.get(i).getNodeIn() != nodeIn) {
            ++i;
        }
        while ((i < this.weights.size()) && (weights.get(i).getNodeIn() == nodeIn)) {
            Weight w = weights.get(i);
            result += error.get(w.getNodeOut()) * w.getValue();
            ++i;
        }
        double hasil = output*(1-output)* result;
        //result = output*(1-output)* result;
        return result;
    }
    
    public double getNewWeight(Weight w, ArrayList<Double> node, ArrayList<Double> error) {
        //System.out.println(this.learning_rate);
        return (w.getValue() + this.learning_rate * node.get(w.getNodeIn()) * error.get(w.getNodeOut()));
    }
    
    @Override
    public void buildClassifier(Instances dataset) throws Exception {
        //Membuat dan inisialisasi weight
        for (int i = 0; i < this.n_in; ++i) {
            int batas_awal;
            int batas_akhir;
            if (this.n_hidden == 0) {
                batas_awal = this.getNIn();
                batas_akhir = this.getTotalN();
            } else {
                batas_awal = this.getNIn()+1;
                batas_akhir = this.n_in + this.n_hidden;
            }
            for (int j = batas_awal; j < batas_akhir; ++j) {
                Weight w = new Weight();
                w.setValue(new Random().nextDouble() * 0.1 - 0.05);
                w.setNodeIn(i);
                w.setNodeOut(j);
                //System.out.println(i + " " + j);
                weights.add(w);    
            }
        }
        
        if (this.n_hidden != 0) {
            for (int i = this.n_in; i < this.n_in + this.n_hidden; ++i) {
                for (int j = this.n_in + this.n_hidden; j < this.getTotalN(); ++j) {
                    Weight w = new Weight();
                    w.setValue(new Random().nextDouble() * 0.1 - 0.05);
                    w.setNodeIn(i);
                    w.setNodeOut(j);
                    weights.add(w);
                }
            }
        }
        
        //
        int loop = 0;
        while (loop < 10000) {
            for (int i = 0; i < dataset.numInstances(); ++i) {
                //Membuat node
                ArrayList<Double> node;
                node = new ArrayList<>();
            
                //-->Forward
                for (int j = 0; j < this.getTotalN(); ++j) {
                    double result;
                    if (j < this.n_in) {
                        if (j == 0) {
                            result = 1.0;
                        } else {
                            result = dataset.instance(i).value(j-1);
                        }
                    } else {
                        if ((j == this.n_in) && (this.n_hidden != 0)) {
                            result = 1.0;
                        } else {
                            result = this.outputValue(this.nettValue(j, node));
                        }
                    }
                    node.add(result);
                }
                
                //<--Backward
                ArrayList<Double> error;
                error = new ArrayList<>();
                for (int j = 0; j < this.getTotalN(); ++j) {
                    error.add(0.00);
                }
                
                //Target
                double target = dataset.instance(i).classValue();
                
                //Output Error
                double temp = this.n_out-1;
                
                for (int j = this.getTotalN()-1; j >= this.n_in + this.n_hidden; --j) {
                    if (temp == target) {
                        error.set(j, errorOutput(node.get(j), 1.0));
                    } else {
                        error.set(j, errorOutput(node.get(j), 0.0));
                    }
                    --temp;
                }
            
                if (this.n_hidden != 0) {
                    for (int j = this.n_in + this.n_hidden - 1; j >= this.n_in; --j) {
                        if (j == this.n_in) {
                            error.set(j, 1.0);
                        } else {
                            error.set(j, this.errorHidden(j, node.get(j), error));
                        }
                    }
                }
            
                //Set nilai baru pada weights
                for (int j = 0; j < weights.size(); ++j) {
                    weights.get(j).setValue(this.getNewWeight(weights.get(j), node, error));
                }
            }
        ++loop;
        }
    }
    
    @Override
    public double classifyInstance(Instance data) throws Exception {
        //Membuat node
        ArrayList<Double> node;
        node = new ArrayList<>();
        
        //-->Forward
        for (int j = 0; j < this.getTotalN(); ++j) {
            double result;
            if (j < this.n_in) {
                if (j == 0) {
                    result = 1.0;
                } else {
                    result = data.value(j-1);
                }
            } else {
                if ((j == this.n_in) && (this.n_hidden != 0)) {
                    result = 1.0;
                } else {
                    result = this.outputValue(this.nettValue(j, node));
                }
            }
            node.add(result);
        }
        
        double result = 0.0;
        if (data.numClasses() > 2) {
            double counter = 0.0;
            double result_class = 0.0;
            double result_output = node.get(this.n_in + this.n_hidden);
            for (int k = this.n_in + this.n_hidden + 1; k < this.getTotalN(); ++k) {
                ++counter;
                if (result_output < node.get(k)) {
                    result_output = node.get(k);
                    result_class = counter;
                }
            }
            return result_class;
        } else {
            if (node.get(node.size()-1) > 0.5) {
                return 0.0;
            } else {
                return 1.0;
            }
        }
    }
}

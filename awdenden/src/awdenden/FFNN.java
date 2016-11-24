
package awdenden;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class FFNN implements Classifier, Serializable {
    public Instances inst;
    public double LEARNING_RATE = 0.1;
    public int neuronInput;
    public int neuronOutput;
    public int nWeights;
    public Node[] inputLayer;
    public Node[] hiddenLayer;
    public Node[] outputLayer;
    public ArrayList<Weight> w = new ArrayList<>();
    
    public FFNN() throws IOException{
        //Instances in;
        
        String filePath = new File("").getAbsolutePath();
        BufferedReader b = new BufferedReader(new FileReader(filePath+"/Team.arff"));
        inst = new Instances(b);
        b.close();
        int classIndex = getIndeksKelas(inst);
        inst.setClassIndex(classIndex);
        neuronInput = inst.numAttributes();
        neuronOutput=inst.numClasses();
        
        inputLayer = new Node[neuronInput];
        hiddenLayer = new Node[neuronInput];
        outputLayer = new Node[neuronOutput+1];
        
        //inisialisasi nodes input dan hidden
        for(int i=0;i<neuronInput; i++){
                inputLayer[i]=new Node(1.0,"x"+Integer.toString(i));
                hiddenLayer[i]=new Node(1.0,"h"+Integer.toString(i));
//                System.out.println(inputLayer[i].getName()+" = "+inputLayer[i].getValue());
//                System.out.println(hiddenLayer[i].getName()+" = "+hiddenLayer[i].getValue());
        }
        
        //inisialisasi nodes output
        for(int i=1;i<=neuronOutput; i++){
            outputLayer[i]=new Node(1.0,"o"+Integer.toString(i));
//            System.out.println(outputLayer[i].getName()+" = "+outputLayer[i].getValue());
        }
        
        //inisialisasi weight dari input ke hidden
        int idx=0;
        for(int i=1; i<neuronInput; i++){
            for(int j=1; j<neuronInput; j++){
                Weight wtemp = new Weight(idx,inputLayer[i], hiddenLayer[j]);
                boolean add = w.add(wtemp);
                System.out.print(idx);
                System.out.print(" "+w.get(idx).getNodeKiri().getName());
                System.out.print(" "+w.get(idx).getNodeKanan().getName());
                System.out.print(" "+w.get(idx).getValue()+"\n");
                idx++;
            }
        }
        
        //inisialisasi weight dari hidden ke output
        for(int i=1; i<neuronInput; i++){
            for(int j=1; j<=neuronOutput; j++){
                Weight wtemp = new Weight(idx,hiddenLayer[i], outputLayer[j]);
                boolean add = w.add(wtemp);
                System.out.print(idx);
                System.out.print(" "+w.get(idx).getNodeKiri().getName());
                System.out.print(" "+w.get(idx).getNodeKanan().getName());
                System.out.print(" "+w.get(idx).getValue()+"\n");
                idx++;
            }
        }
        nWeights=idx+1;
    }
    
    public int getIdxWeight(Node kiri, Node kanan){
        int idx=0;
        for(Weight berat:w){
            if(berat.getNodeKanan().equals(kanan) && berat.getNodeKiri().equals(kiri)){
                return berat.getIdx();
            }
        }
        return idx;
    }
    
    public void setWeight(int idx, double value){
        w.get(idx).setValue(value);
    }
    
    public final int getIndeksKelas(Instances in){
        int classIndex = 0;
        for(int i = 0; i<in.numAttributes(); i++){
            if(in.attribute(i).name().equals("LabelSport")){
                classIndex = i;
                break;
            }
        }
        return classIndex;
    }
    
    public double sigmoid(double x){
        return 1.0/(1.0+Math.exp((-1.0)*x));
    }
    
    public double errorOutput(double target, double output){
        return output * (1-output) * (target-output);
    }
    
    public double errorHidden(double result, double output){
        return output * (1 - output) * result;
    }
    
    public Instances getFiltered(Instances i) throws Exception{
        Instances in;
        Normalize n2b = new Normalize();
        
        n2b.setInputFormat(i);
        in = Filter.useFilter(i, n2b);
        in.setClassIndex(in.numAttributes()-1);
        return in;
    }

    @Override
    public void buildClassifier(Instances i) throws Exception {
       
        
        double[] targetNum = new double[neuronOutput];
        
        int loop = 0;
        while(loop<10000){
            Enumeration<Instance> enumInst = i.enumerateInstances();
            while(enumInst.hasMoreElements()){
                Instance in = enumInst.nextElement();
                for(int j=0; j<neuronOutput; j++){
                    targetNum[j]=0.0;
                }
                targetNum[(int)in.value(i.classIndex())]=1.0;

                //memasukkan value ke input layer
                int j = 1;
                Enumeration<Attribute> enumAtt = i.enumerateAttributes();
                while(enumAtt.hasMoreElements()){
                    Attribute a = enumAtt.nextElement();
                    inputLayer[j].setValue(in.value(a));
                    //System.out.print(in.value(a)+" ");
                    j++;
                }
                //System.out.println();

                //memasukkan value ke hidden layer
                for(j = 1; j<neuronInput; j++){
                    double sigma = 0;
                    for(int k = 0; k<neuronInput; k++){
                        sigma += w.get(getIdxWeight(inputLayer[k],hiddenLayer[j])).getValue()*inputLayer[k].getValue();
                    }
                    hiddenLayer[j].setValue(sigmoid(sigma));
                }

                //memasukkan value ke output layer
                for(j = 1; j<=neuronOutput; j++){
                    double sigma = 0;
                    for(int k = 0; k<neuronInput; k++){
                        sigma+=w.get(getIdxWeight(hiddenLayer[k],outputLayer[j])).getValue()*hiddenLayer[k].getValue();
                    }
                    outputLayer[j].setValue(sigmoid(sigma));
                }

                //mencatat error output
                for(j=1; j<=neuronOutput; j++){
                    outputLayer[j].setError(errorOutput(targetNum[j-1],outputLayer[j].getValue()));
                }

                //mencatat error hidden
                for(j=1; j<neuronInput; j++){
                    double res = 0.0;
                    for(int k = 1; k<=neuronOutput; k++){
                        res += w.get(getIdxWeight(hiddenLayer[j],outputLayer[k])).getValue()*outputLayer[k].getError();
                    }
                    hiddenLayer[j].setError(errorHidden(res,hiddenLayer[j].getValue()));
                }

                //mengupdate weight input ke hidden
                for(j = 0; j<neuronInput; j++){
                    for(int k = 0; k<neuronInput; k++){
                        double wlama = w.get(getIdxWeight(inputLayer[j], hiddenLayer[k])).getValue();
                        double wres = wlama + (LEARNING_RATE*inputLayer[j].getValue()*hiddenLayer[k].getError());
                        w.get(getIdxWeight(inputLayer[j],hiddenLayer[k])).setValue(wres);
                    }
                }

                //mengupdate weight hidden ke input
                for(j = 0; j<neuronInput; j++){
                    for(int k = 1; k<=neuronOutput; k++){
                        double wlama = w.get(getIdxWeight(hiddenLayer[j],outputLayer[k])).getValue();
                        double wres = wlama + (LEARNING_RATE*hiddenLayer[j].getValue()*outputLayer[k].getError());
                        w.get(getIdxWeight(hiddenLayer[j],outputLayer[k])).setValue(wres);
                    }
                }

//                for(Weight wtemp:w){
//                   System.out.println(wtemp.getNodeKiri().getName()+" "+wtemp.getNodeKanan().getName()+" = "+wtemp.getValue());
//                }
//                System.out.println("=============");
            
            }
        System.out.println(loop);
        loop++;
    }
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        int idx = 0;
        double max = -1e9;
        double[] d = distributionForInstance(instnc);
        
        for(int i =0; i<d.length; i++){
            if(d[i]>max){
                idx = i;
                max = d[i];
            }
        }
        
        return idx;
//        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
          double[] d = new double[neuronOutput];
          
          int j = 1;
            Enumeration<Attribute> enumAtt = instnc.enumerateAttributes();
            while(enumAtt.hasMoreElements()){
                Attribute a = enumAtt.nextElement();
                inputLayer[j].setValue(instnc.value(a));
                //System.out.print(in.value(a)+" ");
                j++;
            }
            //System.out.println();
            
            //memasukkan value ke hidden layer
            for(j = 1; j<neuronInput; j++){
                double sigma = 0;
                for(int k = 0; k<neuronInput; k++){
                    sigma += w.get(getIdxWeight(inputLayer[k],hiddenLayer[j])).getValue()*inputLayer[k].getValue();
                }
                hiddenLayer[j].setValue(sigmoid(sigma));
            }
            
            //memasukkan value ke output layer
            for(j = 1; j<=neuronOutput; j++){
                double sigma = 0;
                for(int k = 0; k<neuronInput; k++){
                    sigma+=w.get(getIdxWeight(hiddenLayer[k],outputLayer[j])).getValue()*hiddenLayer[k].getValue();
                }
                outputLayer[j].setValue(sigmoid(sigma));
                d[j-1]=outputLayer[j].getValue();
            }
          System.out.println(Arrays.toString(d));
          return d;
//throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities((CapabilitiesHandler) this);
        result.enableAll();
        return result;
    }
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package awdenden;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 *
 * @author user
 */
public class NaiveBayes008 implements Classifier {
    public Instances inst;
    
    public Instances readInstances() throws IOException{
        String filePath = new File("").getAbsolutePath();
        BufferedReader b = new BufferedReader(new FileReader(filePath+"/mush.arff"));
        inst = new Instances(b);
        int classIndex = getIndeksKelas(inst);
        inst.setClassIndex(classIndex);
        return inst;
    }
    
    public int getIndeksKelas(Instances in){
        int classIndex = 0;
        for(int i = 0; i<inst.numAttributes(); i++){
            if(inst.attribute(i).name().equals("class")){
                classIndex = i;
                break;
            }
        }
        return classIndex;
    }
    
    public Instances getFiltered(Instances i) throws Exception{
        Discretize d = new Discretize();
        Instances in;
        
        d.setInputFormat(i);
        in = Filter.useFilter(i, d);
        in.setClassIndex(getIndeksKelas(in));
        return in;
    }
    

    @Override
    public void buildClassifier(Instances in) throws Exception {
        int totalInstance = in.numInstances();
        System.out.println("Banyak instance: " + totalInstance);
        int banyakAtribut = in.numAttributes()-1;
        System.out.println("Banyak atribut: " + banyakAtribut);
        int jumlahKls = in.numClasses();
        System.out.println("Banyak kelas: "+jumlahKls);
        int[] nKelas = new int[jumlahKls];
        for(int i = 0; i<jumlahKls; i++) nKelas[i]=0;
        
        Enumeration<Attribute> enumAtt = in.enumerateAttributes();
        Enumeration<Instance> enumIns = in.enumerateInstances();
        
        Attribute a = enumAtt.nextElement();
        System.out.println(a.name());
        System.out.println(enumIns.nextElement().value(a));
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}

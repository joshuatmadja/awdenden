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
import java.util.HashMap;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
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
    public LearningMatrix[] public_lm;
    public int[] nKelas = null;
    public double[] probPerKelas = null;
    public HashMap<String, Integer> hm = new HashMap<>();
    
    public NaiveBayes008() throws IOException{
        Instances in;
        String filePath = new File("").getAbsolutePath();
        BufferedReader b = new BufferedReader(new FileReader(filePath+"/mush.arff"));
        inst = new Instances(b);
        int classIndex = getIndeksKelas(inst);
        inst.setClassIndex(classIndex);
    }
    
    public final int getIndeksKelas(Instances in){
        int classIndex = 0;
        for(int i = 0; i<in.numAttributes(); i++){
            if(in.attribute(i).name().equals("class")){
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
        inst = in;
        int totalInstance = in.numInstances();
        System.out.println("Banyak instance: " + totalInstance);
        int banyakAtribut = in.numAttributes()-1;
        System.out.println("Banyak atribut: " + banyakAtribut);
        int jumlahKls = in.numClasses();  
        LearningMatrix[] lm = new LearningMatrix[in.numAttributes()];
        System.out.println("Banyak kelas: "+jumlahKls);
        
        Attribute clAtt = in.classAttribute();
        
        
        //membuat sebuah array nKelas untuk melihat banyak instance untuk kelas itu berapa.
        nKelas = new int[jumlahKls];
        for(int i = 0; i<jumlahKls; i++){
            hm.put(clAtt.value(i), i);
            nKelas[i]=0;
        }
        
        //melihat banyak kelas
        Enumeration<Attribute> enumAtt = in.enumerateAttributes();
        Enumeration<Instance> enumIns = in.enumerateInstances();
        
        while(enumAtt.hasMoreElements()){
            Attribute a = (Attribute) enumAtt.nextElement();
            
            int i = a.index();
            System.out.println("Indeks: "+i+" "+in.numDistinctValues(i));
            lm[i] = new LearningMatrix(a.numValues(),jumlahKls);
          //  System.out.println("Banyak Distinct Value: "+in.numDistinctValues(i));
//            System.out.println(a.numValues());
        }
        
        while(enumIns.hasMoreElements()){
            Instance i = (Instance) enumIns.nextElement();
            int kelas = hm.get(clAtt.value((int) i.value(in.classIndex())));
            //menghitung banyaknya distribusi kelas dalam instances
            nKelas[kelas]++;
            
            for(int j = 0; j<in.numAttributes(); j++){
                if(j!=in.classIndex()){
                    //System.out.println(j+" "+lm[j].getIsi(j,j));
                    
                   lm[j].increase((int) i.value(j), kelas);
                   
                }
//                System.out.print((int) i.value(j)+" ");
            }
//            System.out.println();
        }
        
        probPerKelas = new double[jumlahKls];
        for(int i = 0; i<jumlahKls; i++){
            probPerKelas[i] = (double) (nKelas[i])/in.numInstances();
            System.out.println(probPerKelas[i]);
        }
        
        Enumeration enu = in.enumerateAttributes();
        while(enu.hasMoreElements()){
            Attribute a = (Attribute) enu.nextElement();
            int idx = a.index();
            System.out.println(a.name()+"\n==========");
            //System.out.println("  p  e");
            for(int i = 0; i<lm[idx].getLabel(); i++){
                System.out.print(a.value(i)+" ");
                for(int j=0; j<jumlahKls; j++){
                    double prob = lm[idx].getIsi(i, j)/(double)nKelas[j];
                    lm[idx].setIsi(i, j, prob);
                    if(j!=jumlahKls-1) System.out.print(lm[idx].getIsi(i, j)+" ");
                    else System.out.print(lm[idx].getIsi(i, j)+"\n");
                }
            }
            System.out.println("\n");
        }
        public_lm = lm;
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        //double kelas = 0;
        double[] d = distributionForInstance(instnc);
        
        double max = 0;
        int idx = 0;
        
        for(int i =0; i<d.length; i++){
            if(d[i]>max){
                idx = i;
                max = d[i];
            }
        }
        
//throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        return idx;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        LearningMatrix[] lm = public_lm;
        int jumlahKls = instnc.numClasses();
        Instance in = (Instance) instnc;
        double[] d = new double[instnc.numClasses()];
        for(int i = 0; i<d.length; i++){
            d[i]=1;
        }
        
        
        Enumeration enu = in.enumerateAttributes();
        while(enu.hasMoreElements()){
            Attribute a = (Attribute) enu.nextElement();
            int idx = a.index();
//            System.out.println(idx+" "+in.value(idx));
            System.out.println(a.name()+"\n==========");
            for(int i = 0; i<d.length; i++){
                double kali = lm[idx].getIsi((int) in.value(idx), i);
                d[i]*= kali;
                System.out.println(i+" "+kali);
            }
            
            
            //System.out.println("  p  e");
//            for(int i = 0; i<lm[idx].getLabel(); i++){
//                System.out.print(a.value(i)+" ");
//                for(int j=0; j<jumlahKls; j++){
//                    double prob = lm[idx].getIsi(i, j)/(double)nKelas[j];
//                    lm[idx].setIsi(i, j, prob);
//                    if(j!=jumlahKls-1) System.out.print(lm[idx].getIsi(i, j)+" ");
//                    else System.out.print(lm[idx].getIsi(i, j)+"\n");
//                }
//            }
            System.out.println("\n");
        }
        for(int i=0; i<d.length; i++){
                d[i]*=probPerKelas[i];
                System.out.println("hasil = "+d[i]);
            }
        return d;
//        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities((CapabilitiesHandler) this);
        result.enableAll();
        return result;
//throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}

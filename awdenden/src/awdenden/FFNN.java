
package awdenden;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author user
 */
public class FFNN {
    public Instances iris;
    
    public Instances readInstances() throws FileNotFoundException, IOException, Exception{
        String filePath = new File("").getAbsolutePath();
        BufferedReader b = new BufferedReader(new FileReader (filePath+"/Team.arff"));
        iris = new Instances (b);
        iris.setClassIndex(iris.numAttributes()-1);
        
        return iris;
    }
    
    public Instances getFiltered(Instances i) throws Exception{
        Instances inst;
        NominalToBinary n2b = new NominalToBinary();
        
        n2b.setInputFormat(i);
        inst = Filter.useFilter(i, n2b);
        inst.setClassIndex(inst.numAttributes()-1);
        return inst;
    }
    
    public static void main(String[] args) throws Exception{
        FFNN f = new FFNN();
        f.readInstances();
        Instances jadi = f.getFiltered(f.iris);
        System.out.println("yey");
        System.out.println(jadi);
    }
}

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
import weka.core.Attribute;
import weka.core.Instances;

/**
 *
 * @author user
 */
public class NaiveBayes008 {
    public Instances inst;
    
    public Instances readInstances() throws IOException{
        String filePath = new File("").getAbsolutePath();
        BufferedReader b = new BufferedReader(new FileReader(filePath+"/mush.arff"));
        inst = new Instances(b);
        int classIndex = 0;
        Attribute a;
        for(int i = 0; i<inst.numAttributes(); i++){
            if(inst.attribute(i).name().equals("class")){
                classIndex = i;
                break;
            }
        }
        inst.setClassIndex(classIndex);
        return inst;
    }
    
    public void showAttributes(Instances i){
        System.out.println(i.classAttribute());
    }
}

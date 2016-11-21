/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package awdenden;

import java.io.Serializable;

/**
 *
 * @author user
 */
public class LearningMatrix implements Serializable{
    private double[][] tabel;
    private double label;
    
    public LearningMatrix(int distinctVal, int kelas){
        tabel = new double[distinctVal][kelas];
        for(int i=0; i<distinctVal; i++){
            for(int j = 0; j<kelas; j++){
                tabel[i][j]=0;
            }
        }
        label = distinctVal;
    }
    
    public double getLabel(){
        return label;
    }
    
    public double getIsi(int value, int kelas){
        return tabel[value][kelas];
    }
    
    public void setIsi(int x, int y, double value){
        tabel[x][y]=value;
    }
    
    public void increase(int value, int kelas){
        tabel[value][kelas]++;
    }
    
    public void decrease(int value, int kelas){
        tabel[value][kelas]--;
    }
    
    
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package awdenden;

/**
 *
 * @author user
 */
public class LearningMatrix {
    private int[][] tabel;
    private int label;
    
    public LearningMatrix(int distinctVal, int kelas){
        tabel = new int[distinctVal][kelas];
        for(int i=0; i<distinctVal; i++){
            for(int j = 0; j<kelas; j++){
                tabel[i][j]=0;
            }
        }
        label = distinctVal;
    }
    
    public int getLabel(){
        return label;
    }
    
    public int getIsi(int value, int kelas){
        return tabel[value][kelas];
    }
    
    public void setIsi(int x, int y, int value){
        tabel[x][y]=value;
    }
    
    public void increase(int value, int kelas){
        tabel[value][kelas]++;
    }
    
    public void decrease(int value, int kelas){
        tabel[value][kelas]--;
    }
    
    
}

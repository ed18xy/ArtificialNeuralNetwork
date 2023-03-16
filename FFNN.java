package assign2;

import java.util.ArrayList;
import java.util.Random;

public class FFNN {
    //Multi-layer network
    Layer inputLayer, outputLayer, hiddenLayer;
    ArrayList<Layer> hiddenLayers;
    static Random rnd;

    public FFNN(int num_hidden_layers, int inputLayerSize, int outputLayerSize, 
            ArrayList<Integer> hiddenLayerSize, double biasB, double[] weightsRange, int seed){
        // Initialize network
        hiddenLayers = new ArrayList<>();
        rnd = new Random(seed);
        inputLayer = new Layer(inputLayerSize, 0, weightsRange, hiddenLayerSize.get(0));
        // Initialize multiple hidden layers
        for(int i = 0; i < num_hidden_layers-1; i++){ 
            hiddenLayer = new Layer(hiddenLayerSize.get(i), rnd.nextDouble(), weightsRange, hiddenLayerSize.get(i+1));
            hiddenLayers.add(hiddenLayer);
        }    
        hiddenLayer = new Layer(hiddenLayerSize.get(num_hidden_layers-1), rnd.nextDouble(), weightsRange, outputLayerSize);
        hiddenLayers.add(hiddenLayer);
        outputLayer = new Layer(outputLayerSize, biasB, weightsRange, 0);

    }//FFNN

    //Calculates output of the network based on provided input vector 
    public void forwardPass(ArrayList<Double> inputVector){
        Layer inHidVector;
        inputLayer.assignOutput(inputVector);
        for(int i = 0; i < hiddenLayers.size(); i++){ 
            if(i == 0)inHidVector = inputLayer;
            else inHidVector = hiddenLayers.get(i-1);
            hiddenLayers.get(i).computeLayer(inHidVector);
        }   
        outputLayer.computeLayer(hiddenLayers.get(hiddenLayers.size()-1));
    }//forwardPass

}

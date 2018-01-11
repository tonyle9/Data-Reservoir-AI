package s6regen;

import java.util.Arrays;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.StackPane;
import javafx.scene.text.Text;
import javafx.stage.Stage;

public class Main extends Application {

    public static void main(String[] args) {
       
        
        Reservoir r=new Reservoir(16,64,16,16,16);
        Compute c=new ComputeLayer(r,3);
        r.addComputeUnit(c);
        r.prepareForUse();
        
        float[] in=new float[16];
        Arrays.fill(in, 3.3f);
        r.setInput(in);
        r.computeAll();
        
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Test.");
        StackPane root = new StackPane();
        WritableImage img = new WritableImage(300, 250);
        Canvas canvas = new Canvas(300, 250);
        GraphicsContext gc = canvas.getGraphicsContext2D();
        root.getChildren().add(canvas);
        Text t = new Text("Hello Data Reservoir Compute AI.");
        root.getChildren().add(t);
        primaryStage.setScene(new Scene(root));
        primaryStage.show();
    }

}

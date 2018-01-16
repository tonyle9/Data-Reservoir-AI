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
        AM m = new AM(32, 15);
        float[] t = new float[32];
        float[] s = new float[32];
        for (int k = 0; k < 1000; k++) {
            for (int i = 0; i < 8; i++) {
                t[i] = 1f;
                m.trainVec(t, t);
                t[i] = 0;
            }
        }
        for (int i = 0; i < 8; i++) {
            t[i] = 1f;
            m.recallVec(s, t);
            t[i] = 0;
            System.out.println(Arrays.toString(s));
        }

        Reservoir r = new Reservoir(16, 128, 16, 16 * 4, 16);
        r.addComputeUnit(new ComputeNormalizeInput(r));
        r.addComputeUnit(new ComputeRandomWritable(r, 0));
        r.addComputeUnit(new ComputeAM(r, 3));
        r.addComputeUnit(new ComputeAMWritable(r, 3, 16));
        r.addComputeUnit(new ComputeAMThresholdGated(r, 3, 1f));
        r.addComputeUnit(new ComputeAMThresholdGatedWritable(r, 3, 1f, 16 * 2));
        r.addComputeUnit(new ComputeBinaryWritable(r, 16 * 3));
        r.addComputeUnit(new ComputeLayer(r, 3));
        r.addComputeUnit(new ComputeLayerSq(r, 3));
        r.addComputeUnit(new ComputeLayerFullConSq(r));
        r.addComputeUnit(new ComputeNormalizeGeneral(r));
        r.prepareForUse();

        float[] in = new float[16];
        Arrays.fill(in, 3.3f);
        r.setInput(in);
        r.computeAll();
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

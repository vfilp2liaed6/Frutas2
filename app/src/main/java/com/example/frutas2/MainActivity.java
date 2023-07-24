package com.example.frutas2;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.MediaPlayer;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import com.example.frutas2.ml.Model;

public class MainActivity extends AppCompatActivity {

    Button camara, galeria;
    ImageView imagen;
    TextView resultado;
    int tamañoImagen = 200;
    private Random random = new Random();


    private String[][] enlacesMP3 = {
            {"https://files.catbox.moe/rnmxvu.mp3", "https://files.catbox.moe/5jm5i3.mp3", "https://files.catbox.moe/gj96s1.mp3"}, // Manzana
            {"https://files.catbox.moe/1frstl.mp3", "https://files.catbox.moe/xzsia0.mp3", "https://files.catbox.moe/lq3llq.mp3"}, // Naranja
            {"https://files.catbox.moe/syaiqd.mp3", "https://files.catbox.moe/8sd25o.mp3", "https://files.catbox.moe/sk8g02.mp3"} // Plátano

    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camara = findViewById(R.id.boton);
        galeria = findViewById(R.id.boton2);

        resultado = findViewById(R.id.resultado);
        imagen = findViewById(R.id.imagenView);

        camara.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        galeria.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    public void clasificarImagen(Bitmap imagen){
        try {
            Model modelo = Model.newInstance(getApplicationContext());

            // Crear entradas para referencia.
            TensorBuffer entrada = TensorBuffer.createFixedSize(new int[]{1, 200, 200, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * tamañoImagen * tamañoImagen * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] valoresInt = new int[tamañoImagen * tamañoImagen];
            imagen.getPixels(valoresInt, 0, imagen.getWidth(), 0, 0, imagen.getWidth(), imagen.getHeight());
            int pixel = 0;
            // Iterar sobre cada pixel y extraer los valores R, G, y B. Añadir esos valores individualmente al buffer de bytes.
            for(int i = 0; i < tamañoImagen; i ++){
                for(int j = 0; j < tamañoImagen; j++){
                    int val = valoresInt[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            entrada.loadBuffer(byteBuffer);

            // Ejecutar inferencia del modelo y obtener resultado.
            Model.Outputs salidas = modelo.process(entrada);
            TensorBuffer salida = salidas.getOutputFeature0AsTensorBuffer();

            float[] confianzas = salida.getFloatArray();
            // encontrar el índice de la clase con la confianza más grande.
            int posMax = 0;
            float confianzaMax = 0;
            for (int i = 0; i < confianzas.length; i++) {
                if (confianzas[i] > confianzaMax) {
                    confianzaMax = confianzas[i];
                    posMax = i;
                }
            }

            String[] clases = {"Manzana", "Naranja", "Platano"};

            //String[] clases = {"Platano", "Manzana", "Naranja"};
            resultado.setText(clases[posMax]);

            // Reproducir mp3 seleccionado aleatoriamente para la fruta identificada.
            String urlMP3 = enlacesMP3[posMax][random.nextInt(3)];
            MediaPlayer mediaPlayer = new MediaPlayer();
            mediaPlayer.setDataSource(urlMP3);
            mediaPlayer.prepare(); // Puede tomar tiempo para MediaPlayer descargar y preparar el MP3
            mediaPlayer.start();

            // Liberar recursos del modelo si ya no se utilizan.
            modelo.close();
        } catch (IOException e) {
            // Manejo de excepciones
            Log.e("Clasificación de imagen", "Error en clasificación de imagen", e);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == 1 && data != null && data.getData() != null){
            Uri uri = data.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                Bitmap thumbnail = ThumbnailUtils.extractThumbnail(bitmap, tamañoImagen, tamañoImagen);
                imagen.setImageBitmap(thumbnail);
                clasificarImagen(thumbnail);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if(requestCode == 3 && data != null){
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");
            Bitmap thumbnail = ThumbnailUtils.extractThumbnail(imageBitmap, tamañoImagen, tamañoImagen);
            imagen.setImageBitmap(thumbnail);
            clasificarImagen(thumbnail);
        }
    }
}


class Lingkaran {
    // tuliskan field-field (bidang data)
    //tuliskan metode-metode 
    double radius = 0.0; //sebuah field
    Lingkaran() {

    }
    Lingkaran(double radius) {
        this.radius = radius;
    }
    double dapatluas() {
         return radius * radius * 3.1459;
    }
}
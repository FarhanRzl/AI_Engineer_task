# AI_Engineer_task

Dalam dunia industri, sering kali terjadi kecelakaan kerja yang disebabkan oleh kelalaian manusia, terutama di industri produksi skala besar yang melibatkan mesin-mesin besar dengan potensi bahaya tinggi. Sebagai contoh, pada industri pemotongan kayu, mesin-mesin yang digunakan untuk memotong atau mengolah kayu skala besar dapat menimbulkan risiko fatal jika terjadi kesalahan atau kelalaian manusia.

Pada kasus ini, saya berencana untuk memanfaatkan pendekatan machine learning, khususnya computer vision untuk pengenalan tindakan manusia yang berpotensi membahayakan keselamatan di dunia industri produksi skala besar. Diharapkan, penerapan ini dapat meningkatkan keamanan, efisiensi, dan efektivitas di industri tersebut.

Melalui implementasi pengenalan aksi manusia dalam subtopik computer vision pada machine learning, model diharapkan dapat mengenali aktivitas manusia berdasarkan gerakan tubuh. Untuk pengaplikasiannya, model akan terintegrasi dengan kamera CCTV yang memantau pergerakan manusia secara real-time, dengan hasil rekaman tersebut digunakan sebagai input atau dataset untuk model dalam menentukan aktivitas manusia.

Struktur model dirancang untuk memproses data dengan ringan dan cepat, dengan tujuan untuk memperoleh hasil dalam durasi minimal dan menggunakan sumber daya yang efisien, sehingga aplikasi ini dapat terjangkau dan efektif. Model yang digunakan merupakan pengembangan dari Convolutional Neural Network (CNN), yaitu EfficientNet B0, yang merupakan model pre-trained atau model yang sudah dilatih sebelumnya menggunakan dataset yang sesuai dengan karakter model tersebut. EfficientNet B0 dipilih karena kemampuannya untuk menganalisis fitur dalam frame video, yang dapat menghasilkan akurasi baik dengan penggunaan sumber daya yang ringan. Model ini digunakan sebagai ekstraktor fitur untuk model klasifikasi Long Short-Term Memory (LSTM).

Model LSTM dipilih karena kemampuannya untuk menganalisis hubungan spasio-temporal pada frame yang telah diproses oleh EfficientNet, yang diharapkan dapat meningkatkan akurasi model secara keseluruhan.

Dataset untuk pengujian model ini menggunakan JHMDB dataset, yang terdiri dari 21 kelas aksi manusia. Setiap kelas mewakili aksi manusia yang berbeda, dengan setiap kelas berisi sekitar 40 video dengan durasi 20-40 frame per video.

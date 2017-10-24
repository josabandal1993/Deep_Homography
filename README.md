# Deep_Homography
This project is keras implementtion of <a href="https://arxiv.org/pdf/1606.03798.pdf">Deep Image Homography Estimation</a> paper by DeTone, Malisiewicz and Rabinovich. In the paper the introduced two networks for training, this focused on the implementation of the Regresion Network.
## The Dataset
To generate the dataset used in the paper, download the the <a href="http://cocodataset.org/#download">MSCOCO dataset</a>[train2014, val2014, test2014]. Then run the python data generator codes found in the data_generator folder.
<table style="width:100%">
  <tr>
    <th>Dataset Generated</th>
    <th>Number of image pairs</th>
    <th>Samples per image</th>
  </tr>
  <tr>
    <td>Training</td>
    <td>505,600</td>
    <td>8</td>
  </tr>
  <tr>
    <td>Validation</td>
    <td>6400</td>
    <td>8</td>
  </tr>
  <tr>
    <td>Testing</td>
    <td>5000</td>
    <td>1</td>
  </tr>
</table>

## Training

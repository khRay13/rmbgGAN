import warnings
warnings.simplefilter('ignore')

from model import Utils
from model import Pix2Pix as p2pModel

if __name__ == '__main__':
	row, cdt = Utils().load_data(dirs="Dataset", dataset_name="nobg_mask")
	p2p = p2pModel(dataA=cdt, dataB=row)
	generator = p2p.build_generator()
	discriminator = p2p.build_discriminator()
	p2pgan = p2p.build_gan(generator, discriminator)


	from keras.utils import plot_model
	plot_model(generator, to_file="gen.png", show_shapes="True")
	plot_model(discriminator, to_file="dis.png", show_shapes="True")
	plot_model(p2pgan, to_file="gan.png", show_shapes="True")


	p2p.train(name="NBGM", models=[generator, discriminator, p2pgan], epochs=250, batch_size=16, sample_interval=5)


from lxml import etree


def sum_vehicle_counts(paths):
    total_count = 0
    for xml_file in paths:
        parser = etree.XMLParser(recover=True, remove_comments=True)
        tree = etree.parse(xml_file, parser=parser)
        root = tree.getroot()
        for interval in root.iter("interval"):
            n_veh_contrib = int(interval.get('nVehContrib'))
            total_count += n_veh_contrib
    return total_count


base_dir = "D:/SUMOS/SimpleJunction/"
detectors = ["e1_0", "e1_1", "e1_2", "e1_3", "e1_4", "e1_5", "e1_6", "e1_7"]

# List of XML files containing the detector outputs
xml_files = list(map(lambda x: base_dir + x + '.xml', detectors))
print(sum_vehicle_counts(xml_files))

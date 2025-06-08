import carla
import logging
import numpy as np

class CarlaSimManagerBase:
    def __init__(self, client):
        self.client = client
        self.world = self.client.get_world()
        self.world_map = self.world.get_map()
        self.world_settings = self.world.get_settings()
        self.blueprint_library = self.world.get_blueprint_library()
        self.sensors = {}
        self.listening_sensors = set()

    def spawn_vehicle(self, blueprint_name, spawn_point_index=-1):
        try:
            spawn_points = self.world_map.get_spawn_points()
            if not spawn_points:
                logging.error("Spawn noktası bulunamadı.")
                return None
            if 0 <= spawn_point_index < len(spawn_points):
                transform = spawn_points[spawn_point_index]
            else:
                logging.warning(f"{spawn_point_index}. indeks geçersiz, rasgele spawn noktası kullanılıyor.")
                transform = np.random.choice(spawn_points)

            vehicle_bp = self.blueprint_library.find(blueprint_name)
            if vehicle_bp.id.startswith('vehicle.'):
                vehicle = self.world.spawn_actor(vehicle_bp, transform)
                logging.info(f"Araç oluşturuldu: {vehicle.id}")
                return vehicle
            else:
                logging.error(f"Geçersiz araç blueprint adı: {blueprint_name}")
                return None
        except Exception as e:
            logging.error(f"Araç oluşturulurken bir hata oluştu: {e}")
            return None

    def attach_sensor(self, vehicle, sensor_blueprint_name, attach_transform, callback_function, sensor_config=None):
        if vehicle is None:
            logging.warning("Önce bir araç oluşturulmalı.")
            return None

        sensor_bp = self.blueprint_library.find(sensor_blueprint_name)
        if sensor_bp:
            if sensor_config:
                for key, value in sensor_config.items():
                    try:
                        sensor_bp.set_attribute(key, value)
                    except:
                        logging.warning(f"{sensor_blueprint_name} blueprint'inde '{key}' parametresi bulunamadı.")

            sensor = self.world.spawn_actor(sensor_bp, attach_transform, attach_to=vehicle)
            if sensor:
                self.sensors[sensor.id] = {"actor": sensor, "type": sensor_blueprint_name, "callback": callback_function}
                logging.info(f"{sensor_blueprint_name} sensörü araca eklendi (ID: {sensor.id}).")
                if sensor_config:
                    logging.info(f"{sensor_blueprint_name} sensör parametreleri ayarlandı: {sensor_config}")
                return sensor
            else:
                logging.error(f"{sensor_blueprint_name} sensörü oluşturulamadı.")
                return None
        else:
            logging.error(f"{sensor_blueprint_name} blueprint'i bulunamadı.")
            return None

    def setup_sensors(self, vehicle, sensor_configurations, callback_map):
        for sensor_name, config in sensor_configurations.items():
            callback_name = config.get("callback")
            callback_func = callback_map.get(callback_name)
            if callback_func:
                sensor = self.attach_sensor(
                    vehicle,
                    config["blueprint"],
                    config["transform"],
                    callback_func,
                    config.get("config")
                )
                if sensor:
                    self.sensors[sensor.id]["name"] = sensor_name
            else:
                logging.warning(f"'{callback_name}' adlı geri çağrı fonksiyonu bulunamadı.")

    def destroy_sensors(self):
        logging.info("Sensörler yok ediliyor...")
        if self.sensors:
            for sensor_info in self.sensors.values():
                if sensor_info["actor"] is not None:
                    sensor_info["actor"].destroy()
            self.sensors = {}
            self.listening_sensors = set()
            logging.info("Sensörler yok edildi.")

    def start_all_sensors(self):
        started_count = 0
        for sensor_id, sensor_info in self.sensors.items():
            sensor = sensor_info["actor"]
            callback = sensor_info["callback"]
            if sensor and callback and sensor_id not in self.listening_sensors:
                sensor.listen(callback)
                self.listening_sensors.add(sensor_id)
                started_count += 1
                logging.info(f"Sensör (ID: {sensor_id}, Tip: {sensor_info['type']}, Adı: {sensor_info.get('name', 'Bilinmiyor')}) dinlemeye başladı.")
            elif not callback:
                logging.warning(f"Sensör (ID: {sensor_id}, Adı: {sensor_info.get('name', 'Bilinmiyor')}) için geri çağrı fonksiyonu tanımlanmamış.")
            elif not sensor:
                logging.warning(f"Sensör (ID: {sensor_id}, Adı: {sensor_info.get('name', 'Bilinmiyor')}) geçerli bir aktör değil.")
            elif sensor_id in self.listening_sensors:
                logging.info(f"Sensör (ID: {sensor_id}, Adı: {sensor_info.get('name', 'Bilinmiyor')}) zaten dinliyor.")
        logging.info(f"{started_count} sensör dinlemeye başlatıldı.")

    def stop_all_sensors(self):
        stopped_count = 0
        for sensor_id in list(self.listening_sensors):
            if sensor_id in self.sensors:
                sensor = self.sensors[sensor_id]["actor"]
                if sensor:
                    sensor.stop()
                    self.listening_sensors.remove(sensor_id)
                    stopped_count += 1
                    logging.info(f"Sensör (ID: {sensor_id}, Tip: {self.sensors[sensor_id]['type']}, Adı: {self.sensors[sensor_id].get('name', 'Bilinmiyor')}) dinlemesi durduruldu.")
                else:
                    logging.warning(f"Sensör (ID: {sensor_id}) geçerli bir aktör değil.")
            else:
                logging.warning(f"Dinleyen sensörler listesinde geçersiz sensör ID'si: {sensor_id}")
                self.listening_sensors.remove(sensor_id)
        logging.info(f"{stopped_count} sensörün dinlemesi durduruldu.")
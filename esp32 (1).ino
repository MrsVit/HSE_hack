#include <Arduino.h>
#include <ArduinoJson.h>

// ============================================================
// Модуль 1: полная таблица программ (15 позиций селектора)
// ============================================================
struct WashProgram {
  int id;
  const char* key;
  const char* name;
  const char* operationType;
  const char* fabric;
  int minTemp;
  int maxTemp;
  int maxSpin;
  bool spinFixed;
};

WashProgram programs[] = {
  { 1, "baby_care",      "Детская одежда",    "wash",       "baby",      20,  95, 1000, true  },
  { 2, "shirts_blouses", "Блузки/рубашки",    "wash",       "shirts",    20,  60, 1000, true  },
  { 3, "dark_clothes",   "Тёмная одежда",     "wash",       "dark",      20,  60, 1000, true  },
  { 4, "sportswear",     "Спортивная одежда", "wash",       "sport",     20,  40, 1000, false },
  { 5, "down_jacket",    "Пуховик",           "wash",       "down",      20,  40, 1000, false },
  { 6, "delicate",       "Деликатная стирка", "wash",       "delicate",  20,  40,  600, true  },
  { 7, "cotton",         "Хлопок",            "wash",       "cotton",    20,  95, 1000, true  },
  { 8, "colored_cotton", "Цветной хлопок",    "wash",       "colored",   20,  60, 1000, true  },
  { 9, "synthetic",      "Синтетика",         "wash",       "synthetic", 20,  60, 1000, true  },
  {10, "mixed",          "Смешанная стирка",  "wash",       "mixed",     20,  40, 1000, true  },
  {11, "wool",           "Шерсть",            "wash",       "wool",      20,  40,  800, false },
  {12, "quick_wash",     "Быстрая стирка",    "wash",       "quick",     20,  40,  800, true  },
  {13, "drum_clean",     "Очистка барабана",  "drum_clean", "none",      95,  95,    0, false },
  {14, "rinse_spin",     "Полоскание и отжим","rinse_spin", "none",       0,   0, 1000, false },
  {15, "spin_only",      "Отжим",             "spin_only",  "none",       0,   0, 1000, false },
};
const int PROGRAM_COUNT = sizeof(programs) / sizeof(programs[0]);

// Индекс безопасного режима по умолчанию (Смешанная стирка, id 10)
const int FALLBACK_INDEX = 9; // programs[9] = mixed
// Порог, выше которого результат считается «не найдено»
const int FALLBACK_THRESHOLD = 500;

// ============================================================
// Модуль 2: алгоритм сопоставления
// ============================================================
struct MatchResult {
  int programIndex;   // индекс в массиве programs[], а не id
  int actualSpin;
  bool spinTruncated;
  int score;
};

// Определяем тип операции по запросу модели
const char* detectOperation(const char* requestedOp) {
  if (strcmp(requestedOp, "spin_only") == 0) return "spin_only";
  if (strcmp(requestedOp, "rinse_spin") == 0) return "rinse_spin";
  if (strcmp(requestedOp, "drum_clean") == 0) return "drum_clean";
  return "wash";
}

// Проверяем, является ли ткань «неизвестной»
bool isUnknownFabric(const char* fabric) {
  return (fabric == nullptr) ||
         (strlen(fabric) == 0) ||
         (strcmp(fabric, "unknown") == 0) ||
         (strcmp(fabric, "none") == 0);
}

MatchResult findBestProgram(const char* targetFabric, int targetTemp,
                            int targetSpin, const char* targetOp) {
  MatchResult best = {-1, 0, false, 999999};
  const char* op = detectOperation(targetOp);
  bool fabricUnknown = isUnknownFabric(targetFabric);

  for (int i = 0; i < PROGRAM_COUNT; i++) {
    WashProgram& p = programs[i];

    // 1. Штраф за несовпадение типа операции (подавляющий)
    int opPenalty = (strcmp(op, p.operationType) == 0) ? 0 : 1000;

    // 2. Штраф за несовпадение ткани
    //    Если ткань неизвестна — штраф 0, выбор идёт по температуре/оборотам
    int fabricPenalty;
    if (fabricUnknown || strcmp(op, "wash") != 0) {
      fabricPenalty = 0;
    } else {
      fabricPenalty = (strcmp(targetFabric, p.fabric) == 0) ? 0 : 100;
    }

    // 3. Температура: проверяем попадание в диапазон
    int tempPenalty = 0;
    if (strcmp(op, "wash") == 0 && targetTemp > 0) {
      if (targetTemp < p.minTemp) tempPenalty = p.minTemp - targetTemp;
      else if (targetTemp > p.maxTemp) tempPenalty = targetTemp - p.maxTemp;
    }

    // 4. Обороты
    int actualSpin;
    bool truncated;
    int spinPenalty = 0;

    if (!p.spinFixed) {
      actualSpin = min(targetSpin, p.maxSpin);
      truncated = (targetSpin > p.maxSpin);
      spinPenalty = truncated ? (targetSpin - p.maxSpin) / 10 : 0;
    } else {
      actualSpin = p.maxSpin;
      truncated = (targetSpin > p.maxSpin);
      spinPenalty = abs(targetSpin - p.maxSpin) / 10;
    }

    int totalScore = opPenalty + fabricPenalty + tempPenalty + spinPenalty;

    if (totalScore < best.score) {
      best.programIndex = i;   // сохраняем индекс, а не id
      best.actualSpin = actualSpin;
      best.spinTruncated = truncated;
      best.score = totalScore;
    }
  }

  // Fallback: если ничего разумного не нашли — безопасный режим
  if (best.programIndex < 0 || best.score > FALLBACK_THRESHOLD) {
    WashProgram& fb = programs[FALLBACK_INDEX];
    best.programIndex = FALLBACK_INDEX;
    best.actualSpin = min(targetSpin, fb.maxSpin);
    best.spinTruncated = (targetSpin > fb.maxSpin);
    best.score = -1; // маркер «использован fallback»
  }

  return best;
}

// ============================================================
// Модуль 3: обработка JSON и формирование фактов
// ============================================================
void handleIncomingJson(const String& jsonStr) {
  StaticJsonDocument<384> doc;
  DeserializationError err = deserializeJson(doc, jsonStr);

  if (err) {
    Serial.print("[ERROR] JSON parse failed: ");
    Serial.println(err.c_str());
    return;
  }

  const char* fabric = doc["fabric"] | "cotton";
  int temp = doc["temp"] | 40;
  int spin = doc["spin"] | 800;
  const char* op = doc["operation"] | "wash";

  Serial.print("[INPUT] fabric=");
  Serial.print(fabric);
  Serial.print(", temp=");
  Serial.print(temp);
  Serial.print(", spin=");
  Serial.print(spin);
  Serial.print(", op=");
  Serial.println(op);

  MatchResult result = findBestProgram(fabric, temp, spin, op);

  // Прямой доступ к выбранной программе по индексу — без второго цикла
  WashProgram& chosen = programs[result.programIndex];

  // Только факты, без текстовых message
  StaticJsonDocument<384> resp;
  resp["programId"] = chosen.id;
  resp["programKey"] = chosen.key;
  resp["programName"] = chosen.name;
  resp["operationType"] = chosen.operationType;
  resp["requestedTemp"] = temp;
  resp["programMinTemp"] = chosen.minTemp;
  resp["programMaxTemp"] = chosen.maxTemp;
  resp["requestedSpin"] = spin;
  resp["actualSpin"] = result.actualSpin;
  resp["machineMaxSpin"] = chosen.maxSpin;
  resp["spinFixed"] = chosen.spinFixed;
  resp["truncated"] = result.spinTruncated;
  resp["fallback"] = (result.score == -1); // маркер использованного fallback

  String output;
  serializeJson(resp, output);

  Serial.print("[OUTPUT] ");
  Serial.println(output);
  Serial.println("---");
}

// ============================================================
// Модуль 4: setup / loop
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("=== Laundry Program Matcher (Wokwi Simulation) ===");
  Serial.println("15 programs from Chinese washer manual.");
  Serial.println("Fallback: mixed wash if nothing matches.");
  Serial.println("Unknown fabric: no fabric penalty, temp/spin decide.");
  Serial.println("Send JSON like:");
  Serial.println("  {\"fabric\":\"cotton\",\"temp\":55,\"spin\":1000,\"operation\":\"wash\"}");
  Serial.println("  {\"fabric\":\"alien_material\",\"temp\":40,\"spin\":800}");
  Serial.println("  {\"fabric\":\"unknown\",\"temp\":40,\"spin\":800}");
  Serial.println("==================================================");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      handleIncomingJson(line);
    }
  }
}
# 動きの平滑化を共通基盤から呼ぶ

状態: 採用。2026-10-01に利用者が確認した腕の応答を維持した実装整理。

人体の関節構成とFKは[人体骨格の共通解決ADR](2026-10-02-shared-skeleton-resolution.md)、
腕の安定性は[観測腕ADRの採用済み原則](2026-09-13-observed-arm-tracking.md#採用済みの安定性と維持する原則2026-10-01)
に従う。各処理が同じ減衰式・状態更新を持つと修正が片側にしか届かないため、
`vtuber-tracking::filter`に計算と保持状態をまとめ、各チャンネルから呼び出す。

- `damped`: 既存の臨界減衰式と、scalar・vector・Quaternionの保持状態。
  頭、観測手首・肘位置、肩、肘屈伸、前腕回内/回外が使う。
  掌法線は単位球面の最短回転を解き、同じvector減衰式を使う。
- `exponential`: 時定数と半減期の指数平滑化、最短角度の追従。
  頭位置、translation shaping、胴体位置、視線、上半身の関節角、表情が使う。
  標準表情とARKit表情のattack/release選択も同じparamsのメソッドにまとめる。
  指は[2026-10-03の固定ポーズ選択](2026-10-03-hand-pose-library.md)から、
  確定したポーズへの行き過ぎのない一次補間にも使う。
- `time`: 経過秒・既存の最大dt・遷移進捗・smoothstep。
  頭・位置・表情・腕、欠測のhold/return/acquire、腕のsource遷移、idle補間が使う。

減衰の根拠はDaniel Holdenの公開実装
[Spring-It-On: The Game Developer's Spring-Roll-Call](https://theorangeduck.com/page/spring-roll-call)
のExact Damper、Critical Spring Damper、Quaternion Spring。
一次の時定数は`1-exp(-dt/tau)`、半減期は`1-exp(-ln(2)*dt/half_life)`であり、
同じ秒数でも応答は異なる。二次は既存の逆帯域`omega=1/tau`を保つ。
半減期への設定変換や係数の調整は行わない。

頭の誤差は現在姿勢のlocal tangent、肩はcanonical観測座標のworld residualを使う。
保持する微分状態の座標系も違うため、共通RotationSpringにそれぞれのstepを置く。
肩を頭のlocal方式へ入れ替えたり、頭の観測拒否角を掌へ適用したりしない。
scalar/vectorの保持速度は誤差の微分であり、物理的な移動速度との符号を混同しない。

観測採用、欠測時の保持・初期化、世代変更、反復timestampの扱い、関節制限、
胴体の速度上限は各チャンネルが引き続き所有する。新しい設定や二重フィルタは加えない。
肩・肘・前腕を減衰した後は共通骨格から再構成し、後段の生targetで上書きしない。
source遷移の線形補間と欠測のsmoothstepも既存の形と時間を保つ。

位置の処理段階は[骨格ADRの位置入力](2026-10-02-shared-skeleton-resolution.md#位置入力の段階と仮想腕222)
に従う。体のbridgeが公開したtracked目標を仮想腕が直接読み、胴体用のidle混合後
body-follow出力と区別する。体の追従状態は既存の一回だけ更新し、仮想腕のために
同じ変換や平滑化を再計算しない。既存の目標への補償応答とtorso lagは維持する。

検証は変更クレートの既存テストと、共通計算の30/60/120 Hzの応答、複数軸回転・
座標基底変換・Quaternion符号を確認する。実カメラ・VRM表示・macOSは別の証拠とし、
合成入力の成功を実機目視確認として扱わない。

ローカル確認: tracking lib 321件、avatar lib 374件、arm_ik・arm_motion_geometry・
arm_pose・arm_tracked_mirror・arm_virtual_hand・body_motion_integration・pose_integration・
scheduleの49件、計744件が成功。最後の角度折返し・肘補間の共通化後も該当する
tracked_arm 30件とarm_pipeline 26件が成功。変更2クレートのClippyは成功
（既存avatar警告10件のみ）、fmt・diff checkも成功。
`cargo build -p vtuber-desktop -j 1`で`target/debug/RusTuberV.exe`を
2026-10-02 01:58:45に更新した。修正後の実カメラ映像・VRM目視・macOSは未確認。

var zg=Object.defineProperty;var Eg=(e,t,n)=>t in e?zg(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n;var Ie=(e,t,n)=>Eg(e,typeof t!="symbol"?t+"":t,n);function Pg(e,t){for(var n=0;n<t.length;n++){const r=t[n];if(typeof r!="string"&&!Array.isArray(r)){for(const i in r)if(i!=="default"&&!(i in e)){const o=Object.getOwnPropertyDescriptor(r,i);o&&Object.defineProperty(e,i,o.get?o:{enumerable:!0,get:()=>r[i]})}}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))r(i);new MutationObserver(i=>{for(const o of i)if(o.type==="childList")for(const a of o.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&r(a)}).observe(document,{childList:!0,subtree:!0});function n(i){const o={};return i.integrity&&(o.integrity=i.integrity),i.referrerPolicy&&(o.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?o.credentials="include":i.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function r(i){if(i.ep)return;i.ep=!0;const o=n(i);fetch(i.href,o)}})();var la=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function ar(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}var zh={exports:{}},Oa={},Eh={exports:{}},re={};/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var ro=Symbol.for("react.element"),Ng=Symbol.for("react.portal"),Ag=Symbol.for("react.fragment"),Lg=Symbol.for("react.strict_mode"),jg=Symbol.for("react.profiler"),Rg=Symbol.for("react.provider"),Fg=Symbol.for("react.context"),Ig=Symbol.for("react.forward_ref"),Dg=Symbol.for("react.suspense"),Mg=Symbol.for("react.memo"),qg=Symbol.for("react.lazy"),Hd=Symbol.iterator;function Og(e){return e===null||typeof e!="object"?null:(e=Hd&&e[Hd]||e["@@iterator"],typeof e=="function"?e:null)}var Ph={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},Nh=Object.assign,Ah={};function Yr(e,t,n){this.props=e,this.context=t,this.refs=Ah,this.updater=n||Ph}Yr.prototype.isReactComponent={};Yr.prototype.setState=function(e,t){if(typeof e!="object"&&typeof e!="function"&&e!=null)throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,e,t,"setState")};Yr.prototype.forceUpdate=function(e){this.updater.enqueueForceUpdate(this,e,"forceUpdate")};function Lh(){}Lh.prototype=Yr.prototype;function Su(e,t,n){this.props=e,this.context=t,this.refs=Ah,this.updater=n||Ph}var zu=Su.prototype=new Lh;zu.constructor=Su;Nh(zu,Yr.prototype);zu.isPureReactComponent=!0;var Qd=Array.isArray,jh=Object.prototype.hasOwnProperty,Eu={current:null},Rh={key:!0,ref:!0,__self:!0,__source:!0};function Fh(e,t,n){var r,i={},o=null,a=null;if(t!=null)for(r in t.ref!==void 0&&(a=t.ref),t.key!==void 0&&(o=""+t.key),t)jh.call(t,r)&&!Rh.hasOwnProperty(r)&&(i[r]=t[r]);var s=arguments.length-2;if(s===1)i.children=n;else if(1<s){for(var l=Array(s),u=0;u<s;u++)l[u]=arguments[u+2];i.children=l}if(e&&e.defaultProps)for(r in s=e.defaultProps,s)i[r]===void 0&&(i[r]=s[r]);return{$$typeof:ro,type:e,key:o,ref:a,props:i,_owner:Eu.current}}function Wg(e,t){return{$$typeof:ro,type:e.type,key:t,ref:e.ref,props:e.props,_owner:e._owner}}function Pu(e){return typeof e=="object"&&e!==null&&e.$$typeof===ro}function Vg(e){var t={"=":"=0",":":"=2"};return"$"+e.replace(/[=:]/g,function(n){return t[n]})}var Xd=/\/+/g;function ms(e,t){return typeof e=="object"&&e!==null&&e.key!=null?Vg(""+e.key):t.toString(36)}function Uo(e,t,n,r,i){var o=typeof e;(o==="undefined"||o==="boolean")&&(e=null);var a=!1;if(e===null)a=!0;else switch(o){case"string":case"number":a=!0;break;case"object":switch(e.$$typeof){case ro:case Ng:a=!0}}if(a)return a=e,i=i(a),e=r===""?"."+ms(a,0):r,Qd(i)?(n="",e!=null&&(n=e.replace(Xd,"$&/")+"/"),Uo(i,t,n,"",function(u){return u})):i!=null&&(Pu(i)&&(i=Wg(i,n+(!i.key||a&&a.key===i.key?"":(""+i.key).replace(Xd,"$&/")+"/")+e)),t.push(i)),1;if(a=0,r=r===""?".":r+":",Qd(e))for(var s=0;s<e.length;s++){o=e[s];var l=r+ms(o,s);a+=Uo(o,t,n,l,i)}else if(l=Og(e),typeof l=="function")for(e=l.call(e),s=0;!(o=e.next()).done;)o=o.value,l=r+ms(o,s++),a+=Uo(o,t,n,l,i);else if(o==="object")throw t=String(e),Error("Objects are not valid as a React child (found: "+(t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t)+"). If you meant to render a collection of children, use an array instead.");return a}function yo(e,t,n){if(e==null)return e;var r=[],i=0;return Uo(e,r,"","",function(o){return t.call(n,o,i++)}),r}function Bg(e){if(e._status===-1){var t=e._result;t=t(),t.then(function(n){(e._status===0||e._status===-1)&&(e._status=1,e._result=n)},function(n){(e._status===0||e._status===-1)&&(e._status=2,e._result=n)}),e._status===-1&&(e._status=0,e._result=t)}if(e._status===1)return e._result.default;throw e._result}var Ze={current:null},Ho={transition:null},Ug={ReactCurrentDispatcher:Ze,ReactCurrentBatchConfig:Ho,ReactCurrentOwner:Eu};function Ih(){throw Error("act(...) is not supported in production builds of React.")}re.Children={map:yo,forEach:function(e,t,n){yo(e,function(){t.apply(this,arguments)},n)},count:function(e){var t=0;return yo(e,function(){t++}),t},toArray:function(e){return yo(e,function(t){return t})||[]},only:function(e){if(!Pu(e))throw Error("React.Children.only expected to receive a single React element child.");return e}};re.Component=Yr;re.Fragment=Ag;re.Profiler=jg;re.PureComponent=Su;re.StrictMode=Lg;re.Suspense=Dg;re.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=Ug;re.act=Ih;re.cloneElement=function(e,t,n){if(e==null)throw Error("React.cloneElement(...): The argument must be a React element, but you passed "+e+".");var r=Nh({},e.props),i=e.key,o=e.ref,a=e._owner;if(t!=null){if(t.ref!==void 0&&(o=t.ref,a=Eu.current),t.key!==void 0&&(i=""+t.key),e.type&&e.type.defaultProps)var s=e.type.defaultProps;for(l in t)jh.call(t,l)&&!Rh.hasOwnProperty(l)&&(r[l]=t[l]===void 0&&s!==void 0?s[l]:t[l])}var l=arguments.length-2;if(l===1)r.children=n;else if(1<l){s=Array(l);for(var u=0;u<l;u++)s[u]=arguments[u+2];r.children=s}return{$$typeof:ro,type:e.type,key:i,ref:o,props:r,_owner:a}};re.createContext=function(e){return e={$$typeof:Fg,_currentValue:e,_currentValue2:e,_threadCount:0,Provider:null,Consumer:null,_defaultValue:null,_globalName:null},e.Provider={$$typeof:Rg,_context:e},e.Consumer=e};re.createElement=Fh;re.createFactory=function(e){var t=Fh.bind(null,e);return t.type=e,t};re.createRef=function(){return{current:null}};re.forwardRef=function(e){return{$$typeof:Ig,render:e}};re.isValidElement=Pu;re.lazy=function(e){return{$$typeof:qg,_payload:{_status:-1,_result:e},_init:Bg}};re.memo=function(e,t){return{$$typeof:Mg,type:e,compare:t===void 0?null:t}};re.startTransition=function(e){var t=Ho.transition;Ho.transition={};try{e()}finally{Ho.transition=t}};re.unstable_act=Ih;re.useCallback=function(e,t){return Ze.current.useCallback(e,t)};re.useContext=function(e){return Ze.current.useContext(e)};re.useDebugValue=function(){};re.useDeferredValue=function(e){return Ze.current.useDeferredValue(e)};re.useEffect=function(e,t){return Ze.current.useEffect(e,t)};re.useId=function(){return Ze.current.useId()};re.useImperativeHandle=function(e,t,n){return Ze.current.useImperativeHandle(e,t,n)};re.useInsertionEffect=function(e,t){return Ze.current.useInsertionEffect(e,t)};re.useLayoutEffect=function(e,t){return Ze.current.useLayoutEffect(e,t)};re.useMemo=function(e,t){return Ze.current.useMemo(e,t)};re.useReducer=function(e,t,n){return Ze.current.useReducer(e,t,n)};re.useRef=function(e){return Ze.current.useRef(e)};re.useState=function(e){return Ze.current.useState(e)};re.useSyncExternalStore=function(e,t,n){return Ze.current.useSyncExternalStore(e,t,n)};re.useTransition=function(){return Ze.current.useTransition()};re.version="18.3.1";Eh.exports=re;var C=Eh.exports;const Re=ar(C),Hg=Pg({__proto__:null,default:Re},[C]);/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Qg=C,Xg=Symbol.for("react.element"),Kg=Symbol.for("react.fragment"),Yg=Object.prototype.hasOwnProperty,$g=Qg.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,Gg={key:!0,ref:!0,__self:!0,__source:!0};function Dh(e,t,n){var r,i={},o=null,a=null;n!==void 0&&(o=""+n),t.key!==void 0&&(o=""+t.key),t.ref!==void 0&&(a=t.ref);for(r in t)Yg.call(t,r)&&!Gg.hasOwnProperty(r)&&(i[r]=t[r]);if(e&&e.defaultProps)for(r in t=e.defaultProps,t)i[r]===void 0&&(i[r]=t[r]);return{$$typeof:Xg,type:e,key:o,ref:a,props:i,_owner:$g.current}}Oa.Fragment=Kg;Oa.jsx=Dh;Oa.jsxs=Dh;zh.exports=Oa;var p=zh.exports,pl={},Mh={exports:{}},gt={},qh={exports:{}},Oh={};/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */(function(e){function t(R,U){var b=R.length;R.push(U);e:for(;0<b;){var V=b-1>>>1,H=R[V];if(0<i(H,U))R[V]=U,R[b]=H,b=V;else break e}}function n(R){return R.length===0?null:R[0]}function r(R){if(R.length===0)return null;var U=R[0],b=R.pop();if(b!==U){R[0]=b;e:for(var V=0,H=R.length,v=H>>>1;V<v;){var te=2*(V+1)-1,ce=R[te],se=te+1,et=R[se];if(0>i(ce,b))se<H&&0>i(et,ce)?(R[V]=et,R[se]=b,V=se):(R[V]=ce,R[te]=b,V=te);else if(se<H&&0>i(et,b))R[V]=et,R[se]=b,V=se;else break e}}return U}function i(R,U){var b=R.sortIndex-U.sortIndex;return b!==0?b:R.id-U.id}if(typeof performance=="object"&&typeof performance.now=="function"){var o=performance;e.unstable_now=function(){return o.now()}}else{var a=Date,s=a.now();e.unstable_now=function(){return a.now()-s}}var l=[],u=[],c=1,d=null,h=3,m=!1,f=!1,x=!1,S=typeof setTimeout=="function"?setTimeout:null,g=typeof clearTimeout=="function"?clearTimeout:null,y=typeof setImmediate<"u"?setImmediate:null;typeof navigator<"u"&&navigator.scheduling!==void 0&&navigator.scheduling.isInputPending!==void 0&&navigator.scheduling.isInputPending.bind(navigator.scheduling);function _(R){for(var U=n(u);U!==null;){if(U.callback===null)r(u);else if(U.startTime<=R)r(u),U.sortIndex=U.expirationTime,t(l,U);else break;U=n(u)}}function w(R){if(x=!1,_(R),!f)if(n(l)!==null)f=!0,X(z);else{var U=n(u);U!==null&&ie(w,U.startTime-R)}}function z(R,U){f=!1,x&&(x=!1,g(N),N=-1),m=!0;var b=h;try{for(_(U),d=n(l);d!==null&&(!(d.expirationTime>U)||R&&!D());){var V=d.callback;if(typeof V=="function"){d.callback=null,h=d.priorityLevel;var H=V(d.expirationTime<=U);U=e.unstable_now(),typeof H=="function"?d.callback=H:d===n(l)&&r(l),_(U)}else r(l);d=n(l)}if(d!==null)var v=!0;else{var te=n(u);te!==null&&ie(w,te.startTime-U),v=!1}return v}finally{d=null,h=b,m=!1}}var k=!1,E=null,N=-1,q=5,T=-1;function D(){return!(e.unstable_now()-T<q)}function W(){if(E!==null){var R=e.unstable_now();T=R;var U=!0;try{U=E(!0,R)}finally{U?$():(k=!1,E=null)}}else k=!1}var $;if(typeof y=="function")$=function(){y(W)};else if(typeof MessageChannel<"u"){var Z=new MessageChannel,F=Z.port2;Z.port1.onmessage=W,$=function(){F.postMessage(null)}}else $=function(){S(W,0)};function X(R){E=R,k||(k=!0,$())}function ie(R,U){N=S(function(){R(e.unstable_now())},U)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(R){R.callback=null},e.unstable_continueExecution=function(){f||m||(f=!0,X(z))},e.unstable_forceFrameRate=function(R){0>R||125<R?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):q=0<R?Math.floor(1e3/R):5},e.unstable_getCurrentPriorityLevel=function(){return h},e.unstable_getFirstCallbackNode=function(){return n(l)},e.unstable_next=function(R){switch(h){case 1:case 2:case 3:var U=3;break;default:U=h}var b=h;h=U;try{return R()}finally{h=b}},e.unstable_pauseExecution=function(){},e.unstable_requestPaint=function(){},e.unstable_runWithPriority=function(R,U){switch(R){case 1:case 2:case 3:case 4:case 5:break;default:R=3}var b=h;h=R;try{return U()}finally{h=b}},e.unstable_scheduleCallback=function(R,U,b){var V=e.unstable_now();switch(typeof b=="object"&&b!==null?(b=b.delay,b=typeof b=="number"&&0<b?V+b:V):b=V,R){case 1:var H=-1;break;case 2:H=250;break;case 5:H=1073741823;break;case 4:H=1e4;break;default:H=5e3}return H=b+H,R={id:c++,callback:U,priorityLevel:R,startTime:b,expirationTime:H,sortIndex:-1},b>V?(R.sortIndex=b,t(u,R),n(l)===null&&R===n(u)&&(x?(g(N),N=-1):x=!0,ie(w,b-V))):(R.sortIndex=H,t(l,R),f||m||(f=!0,X(z))),R},e.unstable_shouldYield=D,e.unstable_wrapCallback=function(R){var U=h;return function(){var b=h;h=U;try{return R.apply(this,arguments)}finally{h=b}}}})(Oh);qh.exports=Oh;var Zg=qh.exports;/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Jg=C,ft=Zg;function L(e){for(var t="https://reactjs.org/docs/error-decoder.html?invariant="+e,n=1;n<arguments.length;n++)t+="&args[]="+encodeURIComponent(arguments[n]);return"Minified React error #"+e+"; visit "+t+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}var Wh=new Set,Ri={};function sr(e,t){Or(e,t),Or(e+"Capture",t)}function Or(e,t){for(Ri[e]=t,e=0;e<t.length;e++)Wh.add(t[e])}var on=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),hl=Object.prototype.hasOwnProperty,e_=/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,Kd={},Yd={};function t_(e){return hl.call(Yd,e)?!0:hl.call(Kd,e)?!1:e_.test(e)?Yd[e]=!0:(Kd[e]=!0,!1)}function n_(e,t,n,r){if(n!==null&&n.type===0)return!1;switch(typeof t){case"function":case"symbol":return!0;case"boolean":return r?!1:n!==null?!n.acceptsBooleans:(e=e.toLowerCase().slice(0,5),e!=="data-"&&e!=="aria-");default:return!1}}function r_(e,t,n,r){if(t===null||typeof t>"u"||n_(e,t,n,r))return!0;if(r)return!1;if(n!==null)switch(n.type){case 3:return!t;case 4:return t===!1;case 5:return isNaN(t);case 6:return isNaN(t)||1>t}return!1}function Je(e,t,n,r,i,o,a){this.acceptsBooleans=t===2||t===3||t===4,this.attributeName=r,this.attributeNamespace=i,this.mustUseProperty=n,this.propertyName=e,this.type=t,this.sanitizeURL=o,this.removeEmptyString=a}var qe={};"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e){qe[e]=new Je(e,0,!1,e,null,!1,!1)});[["acceptCharset","accept-charset"],["className","class"],["htmlFor","for"],["httpEquiv","http-equiv"]].forEach(function(e){var t=e[0];qe[t]=new Je(t,1,!1,e[1],null,!1,!1)});["contentEditable","draggable","spellCheck","value"].forEach(function(e){qe[e]=new Je(e,2,!1,e.toLowerCase(),null,!1,!1)});["autoReverse","externalResourcesRequired","focusable","preserveAlpha"].forEach(function(e){qe[e]=new Je(e,2,!1,e,null,!1,!1)});"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e){qe[e]=new Je(e,3,!1,e.toLowerCase(),null,!1,!1)});["checked","multiple","muted","selected"].forEach(function(e){qe[e]=new Je(e,3,!0,e,null,!1,!1)});["capture","download"].forEach(function(e){qe[e]=new Je(e,4,!1,e,null,!1,!1)});["cols","rows","size","span"].forEach(function(e){qe[e]=new Je(e,6,!1,e,null,!1,!1)});["rowSpan","start"].forEach(function(e){qe[e]=new Je(e,5,!1,e.toLowerCase(),null,!1,!1)});var Nu=/[\-:]([a-z])/g;function Au(e){return e[1].toUpperCase()}"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e){var t=e.replace(Nu,Au);qe[t]=new Je(t,1,!1,e,null,!1,!1)});"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e){var t=e.replace(Nu,Au);qe[t]=new Je(t,1,!1,e,"http://www.w3.org/1999/xlink",!1,!1)});["xml:base","xml:lang","xml:space"].forEach(function(e){var t=e.replace(Nu,Au);qe[t]=new Je(t,1,!1,e,"http://www.w3.org/XML/1998/namespace",!1,!1)});["tabIndex","crossOrigin"].forEach(function(e){qe[e]=new Je(e,1,!1,e.toLowerCase(),null,!1,!1)});qe.xlinkHref=new Je("xlinkHref",1,!1,"xlink:href","http://www.w3.org/1999/xlink",!0,!1);["src","href","action","formAction"].forEach(function(e){qe[e]=new Je(e,1,!1,e.toLowerCase(),null,!0,!0)});function Lu(e,t,n,r){var i=qe.hasOwnProperty(t)?qe[t]:null;(i!==null?i.type!==0:r||!(2<t.length)||t[0]!=="o"&&t[0]!=="O"||t[1]!=="n"&&t[1]!=="N")&&(r_(t,n,i,r)&&(n=null),r||i===null?t_(t)&&(n===null?e.removeAttribute(t):e.setAttribute(t,""+n)):i.mustUseProperty?e[i.propertyName]=n===null?i.type===3?!1:"":n:(t=i.attributeName,r=i.attributeNamespace,n===null?e.removeAttribute(t):(i=i.type,n=i===3||i===4&&n===!0?"":""+n,r?e.setAttributeNS(r,t,n):e.setAttribute(t,n))))}var un=Jg.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,xo=Symbol.for("react.element"),_r=Symbol.for("react.portal"),yr=Symbol.for("react.fragment"),ju=Symbol.for("react.strict_mode"),ml=Symbol.for("react.profiler"),Vh=Symbol.for("react.provider"),Bh=Symbol.for("react.context"),Ru=Symbol.for("react.forward_ref"),fl=Symbol.for("react.suspense"),gl=Symbol.for("react.suspense_list"),Fu=Symbol.for("react.memo"),yn=Symbol.for("react.lazy"),Uh=Symbol.for("react.offscreen"),$d=Symbol.iterator;function ri(e){return e===null||typeof e!="object"?null:(e=$d&&e[$d]||e["@@iterator"],typeof e=="function"?e:null)}var we=Object.assign,fs;function gi(e){if(fs===void 0)try{throw Error()}catch(n){var t=n.stack.trim().match(/\n( *(at )?)/);fs=t&&t[1]||""}return`
`+fs+e}var gs=!1;function _s(e,t){if(!e||gs)return"";gs=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{if(t)if(t=function(){throw Error()},Object.defineProperty(t.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(t,[])}catch(u){var r=u}Reflect.construct(e,[],t)}else{try{t.call()}catch(u){r=u}e.call(t.prototype)}else{try{throw Error()}catch(u){r=u}e()}}catch(u){if(u&&r&&typeof u.stack=="string"){for(var i=u.stack.split(`
`),o=r.stack.split(`
`),a=i.length-1,s=o.length-1;1<=a&&0<=s&&i[a]!==o[s];)s--;for(;1<=a&&0<=s;a--,s--)if(i[a]!==o[s]){if(a!==1||s!==1)do if(a--,s--,0>s||i[a]!==o[s]){var l=`
`+i[a].replace(" at new "," at ");return e.displayName&&l.includes("<anonymous>")&&(l=l.replace("<anonymous>",e.displayName)),l}while(1<=a&&0<=s);break}}}finally{gs=!1,Error.prepareStackTrace=n}return(e=e?e.displayName||e.name:"")?gi(e):""}function i_(e){switch(e.tag){case 5:return gi(e.type);case 16:return gi("Lazy");case 13:return gi("Suspense");case 19:return gi("SuspenseList");case 0:case 2:case 15:return e=_s(e.type,!1),e;case 11:return e=_s(e.type.render,!1),e;case 1:return e=_s(e.type,!0),e;default:return""}}function _l(e){if(e==null)return null;if(typeof e=="function")return e.displayName||e.name||null;if(typeof e=="string")return e;switch(e){case yr:return"Fragment";case _r:return"Portal";case ml:return"Profiler";case ju:return"StrictMode";case fl:return"Suspense";case gl:return"SuspenseList"}if(typeof e=="object")switch(e.$$typeof){case Bh:return(e.displayName||"Context")+".Consumer";case Vh:return(e._context.displayName||"Context")+".Provider";case Ru:var t=e.render;return e=e.displayName,e||(e=t.displayName||t.name||"",e=e!==""?"ForwardRef("+e+")":"ForwardRef"),e;case Fu:return t=e.displayName||null,t!==null?t:_l(e.type)||"Memo";case yn:t=e._payload,e=e._init;try{return _l(e(t))}catch{}}return null}function o_(e){var t=e.type;switch(e.tag){case 24:return"Cache";case 9:return(t.displayName||"Context")+".Consumer";case 10:return(t._context.displayName||"Context")+".Provider";case 18:return"DehydratedFragment";case 11:return e=t.render,e=e.displayName||e.name||"",t.displayName||(e!==""?"ForwardRef("+e+")":"ForwardRef");case 7:return"Fragment";case 5:return t;case 4:return"Portal";case 3:return"Root";case 6:return"Text";case 16:return _l(t);case 8:return t===ju?"StrictMode":"Mode";case 22:return"Offscreen";case 12:return"Profiler";case 21:return"Scope";case 13:return"Suspense";case 19:return"SuspenseList";case 25:return"TracingMarker";case 1:case 0:case 17:case 2:case 14:case 15:if(typeof t=="function")return t.displayName||t.name||null;if(typeof t=="string")return t}return null}function In(e){switch(typeof e){case"boolean":case"number":case"string":case"undefined":return e;case"object":return e;default:return""}}function Hh(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()==="input"&&(t==="checkbox"||t==="radio")}function a_(e){var t=Hh(e)?"checked":"value",n=Object.getOwnPropertyDescriptor(e.constructor.prototype,t),r=""+e[t];if(!e.hasOwnProperty(t)&&typeof n<"u"&&typeof n.get=="function"&&typeof n.set=="function"){var i=n.get,o=n.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return i.call(this)},set:function(a){r=""+a,o.call(this,a)}}),Object.defineProperty(e,t,{enumerable:n.enumerable}),{getValue:function(){return r},setValue:function(a){r=""+a},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function bo(e){e._valueTracker||(e._valueTracker=a_(e))}function Qh(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),r="";return e&&(r=Hh(e)?e.checked?"true":"false":e.value),e=r,e!==n?(t.setValue(e),!0):!1}function ua(e){if(e=e||(typeof document<"u"?document:void 0),typeof e>"u")return null;try{return e.activeElement||e.body}catch{return e.body}}function yl(e,t){var n=t.checked;return we({},t,{defaultChecked:void 0,defaultValue:void 0,value:void 0,checked:n??e._wrapperState.initialChecked})}function Gd(e,t){var n=t.defaultValue==null?"":t.defaultValue,r=t.checked!=null?t.checked:t.defaultChecked;n=In(t.value!=null?t.value:n),e._wrapperState={initialChecked:r,initialValue:n,controlled:t.type==="checkbox"||t.type==="radio"?t.checked!=null:t.value!=null}}function Xh(e,t){t=t.checked,t!=null&&Lu(e,"checked",t,!1)}function xl(e,t){Xh(e,t);var n=In(t.value),r=t.type;if(n!=null)r==="number"?(n===0&&e.value===""||e.value!=n)&&(e.value=""+n):e.value!==""+n&&(e.value=""+n);else if(r==="submit"||r==="reset"){e.removeAttribute("value");return}t.hasOwnProperty("value")?bl(e,t.type,n):t.hasOwnProperty("defaultValue")&&bl(e,t.type,In(t.defaultValue)),t.checked==null&&t.defaultChecked!=null&&(e.defaultChecked=!!t.defaultChecked)}function Zd(e,t,n){if(t.hasOwnProperty("value")||t.hasOwnProperty("defaultValue")){var r=t.type;if(!(r!=="submit"&&r!=="reset"||t.value!==void 0&&t.value!==null))return;t=""+e._wrapperState.initialValue,n||t===e.value||(e.value=t),e.defaultValue=t}n=e.name,n!==""&&(e.name=""),e.defaultChecked=!!e._wrapperState.initialChecked,n!==""&&(e.name=n)}function bl(e,t,n){(t!=="number"||ua(e.ownerDocument)!==e)&&(n==null?e.defaultValue=""+e._wrapperState.initialValue:e.defaultValue!==""+n&&(e.defaultValue=""+n))}var _i=Array.isArray;function Nr(e,t,n,r){if(e=e.options,t){t={};for(var i=0;i<n.length;i++)t["$"+n[i]]=!0;for(n=0;n<e.length;n++)i=t.hasOwnProperty("$"+e[n].value),e[n].selected!==i&&(e[n].selected=i),i&&r&&(e[n].defaultSelected=!0)}else{for(n=""+In(n),t=null,i=0;i<e.length;i++){if(e[i].value===n){e[i].selected=!0,r&&(e[i].defaultSelected=!0);return}t!==null||e[i].disabled||(t=e[i])}t!==null&&(t.selected=!0)}}function vl(e,t){if(t.dangerouslySetInnerHTML!=null)throw Error(L(91));return we({},t,{value:void 0,defaultValue:void 0,children:""+e._wrapperState.initialValue})}function Jd(e,t){var n=t.value;if(n==null){if(n=t.children,t=t.defaultValue,n!=null){if(t!=null)throw Error(L(92));if(_i(n)){if(1<n.length)throw Error(L(93));n=n[0]}t=n}t==null&&(t=""),n=t}e._wrapperState={initialValue:In(n)}}function Kh(e,t){var n=In(t.value),r=In(t.defaultValue);n!=null&&(n=""+n,n!==e.value&&(e.value=n),t.defaultValue==null&&e.defaultValue!==n&&(e.defaultValue=n)),r!=null&&(e.defaultValue=""+r)}function ec(e){var t=e.textContent;t===e._wrapperState.initialValue&&t!==""&&t!==null&&(e.value=t)}function Yh(e){switch(e){case"svg":return"http://www.w3.org/2000/svg";case"math":return"http://www.w3.org/1998/Math/MathML";default:return"http://www.w3.org/1999/xhtml"}}function kl(e,t){return e==null||e==="http://www.w3.org/1999/xhtml"?Yh(t):e==="http://www.w3.org/2000/svg"&&t==="foreignObject"?"http://www.w3.org/1999/xhtml":e}var vo,$h=function(e){return typeof MSApp<"u"&&MSApp.execUnsafeLocalFunction?function(t,n,r,i){MSApp.execUnsafeLocalFunction(function(){return e(t,n,r,i)})}:e}(function(e,t){if(e.namespaceURI!=="http://www.w3.org/2000/svg"||"innerHTML"in e)e.innerHTML=t;else{for(vo=vo||document.createElement("div"),vo.innerHTML="<svg>"+t.valueOf().toString()+"</svg>",t=vo.firstChild;e.firstChild;)e.removeChild(e.firstChild);for(;t.firstChild;)e.appendChild(t.firstChild)}});function Fi(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var vi={animationIterationCount:!0,aspectRatio:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridArea:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},s_=["Webkit","ms","Moz","O"];Object.keys(vi).forEach(function(e){s_.forEach(function(t){t=t+e.charAt(0).toUpperCase()+e.substring(1),vi[t]=vi[e]})});function Gh(e,t,n){return t==null||typeof t=="boolean"||t===""?"":n||typeof t!="number"||t===0||vi.hasOwnProperty(e)&&vi[e]?(""+t).trim():t+"px"}function Zh(e,t){e=e.style;for(var n in t)if(t.hasOwnProperty(n)){var r=n.indexOf("--")===0,i=Gh(n,t[n],r);n==="float"&&(n="cssFloat"),r?e.setProperty(n,i):e[n]=i}}var l_=we({menuitem:!0},{area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0});function wl(e,t){if(t){if(l_[e]&&(t.children!=null||t.dangerouslySetInnerHTML!=null))throw Error(L(137,e));if(t.dangerouslySetInnerHTML!=null){if(t.children!=null)throw Error(L(60));if(typeof t.dangerouslySetInnerHTML!="object"||!("__html"in t.dangerouslySetInnerHTML))throw Error(L(61))}if(t.style!=null&&typeof t.style!="object")throw Error(L(62))}}function Tl(e,t){if(e.indexOf("-")===-1)return typeof t.is=="string";switch(e){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var Cl=null;function Iu(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var Sl=null,Ar=null,Lr=null;function tc(e){if(e=ao(e)){if(typeof Sl!="function")throw Error(L(280));var t=e.stateNode;t&&(t=Ha(t),Sl(e.stateNode,e.type,t))}}function Jh(e){Ar?Lr?Lr.push(e):Lr=[e]:Ar=e}function em(){if(Ar){var e=Ar,t=Lr;if(Lr=Ar=null,tc(e),t)for(e=0;e<t.length;e++)tc(t[e])}}function tm(e,t){return e(t)}function nm(){}var ys=!1;function rm(e,t,n){if(ys)return e(t,n);ys=!0;try{return tm(e,t,n)}finally{ys=!1,(Ar!==null||Lr!==null)&&(nm(),em())}}function Ii(e,t){var n=e.stateNode;if(n===null)return null;var r=Ha(n);if(r===null)return null;n=r[t];e:switch(t){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(r=!r.disabled)||(e=e.type,r=!(e==="button"||e==="input"||e==="select"||e==="textarea")),e=!r;break e;default:e=!1}if(e)return null;if(n&&typeof n!="function")throw Error(L(231,t,typeof n));return n}var zl=!1;if(on)try{var ii={};Object.defineProperty(ii,"passive",{get:function(){zl=!0}}),window.addEventListener("test",ii,ii),window.removeEventListener("test",ii,ii)}catch{zl=!1}function u_(e,t,n,r,i,o,a,s,l){var u=Array.prototype.slice.call(arguments,3);try{t.apply(n,u)}catch(c){this.onError(c)}}var ki=!1,da=null,ca=!1,El=null,d_={onError:function(e){ki=!0,da=e}};function c_(e,t,n,r,i,o,a,s,l){ki=!1,da=null,u_.apply(d_,arguments)}function p_(e,t,n,r,i,o,a,s,l){if(c_.apply(this,arguments),ki){if(ki){var u=da;ki=!1,da=null}else throw Error(L(198));ca||(ca=!0,El=u)}}function lr(e){var t=e,n=e;if(e.alternate)for(;t.return;)t=t.return;else{e=t;do t=e,t.flags&4098&&(n=t.return),e=t.return;while(e)}return t.tag===3?n:null}function im(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function nc(e){if(lr(e)!==e)throw Error(L(188))}function h_(e){var t=e.alternate;if(!t){if(t=lr(e),t===null)throw Error(L(188));return t!==e?null:e}for(var n=e,r=t;;){var i=n.return;if(i===null)break;var o=i.alternate;if(o===null){if(r=i.return,r!==null){n=r;continue}break}if(i.child===o.child){for(o=i.child;o;){if(o===n)return nc(i),e;if(o===r)return nc(i),t;o=o.sibling}throw Error(L(188))}if(n.return!==r.return)n=i,r=o;else{for(var a=!1,s=i.child;s;){if(s===n){a=!0,n=i,r=o;break}if(s===r){a=!0,r=i,n=o;break}s=s.sibling}if(!a){for(s=o.child;s;){if(s===n){a=!0,n=o,r=i;break}if(s===r){a=!0,r=o,n=i;break}s=s.sibling}if(!a)throw Error(L(189))}}if(n.alternate!==r)throw Error(L(190))}if(n.tag!==3)throw Error(L(188));return n.stateNode.current===n?e:t}function om(e){return e=h_(e),e!==null?am(e):null}function am(e){if(e.tag===5||e.tag===6)return e;for(e=e.child;e!==null;){var t=am(e);if(t!==null)return t;e=e.sibling}return null}var sm=ft.unstable_scheduleCallback,rc=ft.unstable_cancelCallback,m_=ft.unstable_shouldYield,f_=ft.unstable_requestPaint,Se=ft.unstable_now,g_=ft.unstable_getCurrentPriorityLevel,Du=ft.unstable_ImmediatePriority,lm=ft.unstable_UserBlockingPriority,pa=ft.unstable_NormalPriority,__=ft.unstable_LowPriority,um=ft.unstable_IdlePriority,Wa=null,Xt=null;function y_(e){if(Xt&&typeof Xt.onCommitFiberRoot=="function")try{Xt.onCommitFiberRoot(Wa,e,void 0,(e.current.flags&128)===128)}catch{}}var Mt=Math.clz32?Math.clz32:v_,x_=Math.log,b_=Math.LN2;function v_(e){return e>>>=0,e===0?32:31-(x_(e)/b_|0)|0}var ko=64,wo=4194304;function yi(e){switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e&4194240;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return e&130023424;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 1073741824;default:return e}}function ha(e,t){var n=e.pendingLanes;if(n===0)return 0;var r=0,i=e.suspendedLanes,o=e.pingedLanes,a=n&268435455;if(a!==0){var s=a&~i;s!==0?r=yi(s):(o&=a,o!==0&&(r=yi(o)))}else a=n&~i,a!==0?r=yi(a):o!==0&&(r=yi(o));if(r===0)return 0;if(t!==0&&t!==r&&!(t&i)&&(i=r&-r,o=t&-t,i>=o||i===16&&(o&4194240)!==0))return t;if(r&4&&(r|=n&16),t=e.entangledLanes,t!==0)for(e=e.entanglements,t&=r;0<t;)n=31-Mt(t),i=1<<n,r|=e[n],t&=~i;return r}function k_(e,t){switch(e){case 1:case 2:case 4:return t+250;case 8:case 16:case 32:case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return-1;case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function w_(e,t){for(var n=e.suspendedLanes,r=e.pingedLanes,i=e.expirationTimes,o=e.pendingLanes;0<o;){var a=31-Mt(o),s=1<<a,l=i[a];l===-1?(!(s&n)||s&r)&&(i[a]=k_(s,t)):l<=t&&(e.expiredLanes|=s),o&=~s}}function Pl(e){return e=e.pendingLanes&-1073741825,e!==0?e:e&1073741824?1073741824:0}function dm(){var e=ko;return ko<<=1,!(ko&4194240)&&(ko=64),e}function xs(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function io(e,t,n){e.pendingLanes|=t,t!==536870912&&(e.suspendedLanes=0,e.pingedLanes=0),e=e.eventTimes,t=31-Mt(t),e[t]=n}function T_(e,t){var n=e.pendingLanes&~t;e.pendingLanes=t,e.suspendedLanes=0,e.pingedLanes=0,e.expiredLanes&=t,e.mutableReadLanes&=t,e.entangledLanes&=t,t=e.entanglements;var r=e.eventTimes;for(e=e.expirationTimes;0<n;){var i=31-Mt(n),o=1<<i;t[i]=0,r[i]=-1,e[i]=-1,n&=~o}}function Mu(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var r=31-Mt(n),i=1<<r;i&t|e[r]&t&&(e[r]|=t),n&=~i}}var de=0;function cm(e){return e&=-e,1<e?4<e?e&268435455?16:536870912:4:1}var pm,qu,hm,mm,fm,Nl=!1,To=[],Sn=null,zn=null,En=null,Di=new Map,Mi=new Map,bn=[],C_="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");function ic(e,t){switch(e){case"focusin":case"focusout":Sn=null;break;case"dragenter":case"dragleave":zn=null;break;case"mouseover":case"mouseout":En=null;break;case"pointerover":case"pointerout":Di.delete(t.pointerId);break;case"gotpointercapture":case"lostpointercapture":Mi.delete(t.pointerId)}}function oi(e,t,n,r,i,o){return e===null||e.nativeEvent!==o?(e={blockedOn:t,domEventName:n,eventSystemFlags:r,nativeEvent:o,targetContainers:[i]},t!==null&&(t=ao(t),t!==null&&qu(t)),e):(e.eventSystemFlags|=r,t=e.targetContainers,i!==null&&t.indexOf(i)===-1&&t.push(i),e)}function S_(e,t,n,r,i){switch(t){case"focusin":return Sn=oi(Sn,e,t,n,r,i),!0;case"dragenter":return zn=oi(zn,e,t,n,r,i),!0;case"mouseover":return En=oi(En,e,t,n,r,i),!0;case"pointerover":var o=i.pointerId;return Di.set(o,oi(Di.get(o)||null,e,t,n,r,i)),!0;case"gotpointercapture":return o=i.pointerId,Mi.set(o,oi(Mi.get(o)||null,e,t,n,r,i)),!0}return!1}function gm(e){var t=Kn(e.target);if(t!==null){var n=lr(t);if(n!==null){if(t=n.tag,t===13){if(t=im(n),t!==null){e.blockedOn=t,fm(e.priority,function(){hm(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function Qo(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=Al(e.domEventName,e.eventSystemFlags,t[0],e.nativeEvent);if(n===null){n=e.nativeEvent;var r=new n.constructor(n.type,n);Cl=r,n.target.dispatchEvent(r),Cl=null}else return t=ao(n),t!==null&&qu(t),e.blockedOn=n,!1;t.shift()}return!0}function oc(e,t,n){Qo(e)&&n.delete(t)}function z_(){Nl=!1,Sn!==null&&Qo(Sn)&&(Sn=null),zn!==null&&Qo(zn)&&(zn=null),En!==null&&Qo(En)&&(En=null),Di.forEach(oc),Mi.forEach(oc)}function ai(e,t){e.blockedOn===t&&(e.blockedOn=null,Nl||(Nl=!0,ft.unstable_scheduleCallback(ft.unstable_NormalPriority,z_)))}function qi(e){function t(i){return ai(i,e)}if(0<To.length){ai(To[0],e);for(var n=1;n<To.length;n++){var r=To[n];r.blockedOn===e&&(r.blockedOn=null)}}for(Sn!==null&&ai(Sn,e),zn!==null&&ai(zn,e),En!==null&&ai(En,e),Di.forEach(t),Mi.forEach(t),n=0;n<bn.length;n++)r=bn[n],r.blockedOn===e&&(r.blockedOn=null);for(;0<bn.length&&(n=bn[0],n.blockedOn===null);)gm(n),n.blockedOn===null&&bn.shift()}var jr=un.ReactCurrentBatchConfig,ma=!0;function E_(e,t,n,r){var i=de,o=jr.transition;jr.transition=null;try{de=1,Ou(e,t,n,r)}finally{de=i,jr.transition=o}}function P_(e,t,n,r){var i=de,o=jr.transition;jr.transition=null;try{de=4,Ou(e,t,n,r)}finally{de=i,jr.transition=o}}function Ou(e,t,n,r){if(ma){var i=Al(e,t,n,r);if(i===null)Ps(e,t,r,fa,n),ic(e,r);else if(S_(i,e,t,n,r))r.stopPropagation();else if(ic(e,r),t&4&&-1<C_.indexOf(e)){for(;i!==null;){var o=ao(i);if(o!==null&&pm(o),o=Al(e,t,n,r),o===null&&Ps(e,t,r,fa,n),o===i)break;i=o}i!==null&&r.stopPropagation()}else Ps(e,t,r,null,n)}}var fa=null;function Al(e,t,n,r){if(fa=null,e=Iu(r),e=Kn(e),e!==null)if(t=lr(e),t===null)e=null;else if(n=t.tag,n===13){if(e=im(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null);return fa=e,null}function _m(e){switch(e){case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 1;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"toggle":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 4;case"message":switch(g_()){case Du:return 1;case lm:return 4;case pa:case __:return 16;case um:return 536870912;default:return 16}default:return 16}}var kn=null,Wu=null,Xo=null;function ym(){if(Xo)return Xo;var e,t=Wu,n=t.length,r,i="value"in kn?kn.value:kn.textContent,o=i.length;for(e=0;e<n&&t[e]===i[e];e++);var a=n-e;for(r=1;r<=a&&t[n-r]===i[o-r];r++);return Xo=i.slice(e,1<r?1-r:void 0)}function Ko(e){var t=e.keyCode;return"charCode"in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function Co(){return!0}function ac(){return!1}function _t(e){function t(n,r,i,o,a){this._reactName=n,this._targetInst=i,this.type=r,this.nativeEvent=o,this.target=a,this.currentTarget=null;for(var s in e)e.hasOwnProperty(s)&&(n=e[s],this[s]=n?n(o):o[s]);return this.isDefaultPrevented=(o.defaultPrevented!=null?o.defaultPrevented:o.returnValue===!1)?Co:ac,this.isPropagationStopped=ac,this}return we(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=Co)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=Co)},persist:function(){},isPersistent:Co}),t}var $r={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},Vu=_t($r),oo=we({},$r,{view:0,detail:0}),N_=_t(oo),bs,vs,si,Va=we({},oo,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:Bu,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return"movementX"in e?e.movementX:(e!==si&&(si&&e.type==="mousemove"?(bs=e.screenX-si.screenX,vs=e.screenY-si.screenY):vs=bs=0,si=e),bs)},movementY:function(e){return"movementY"in e?e.movementY:vs}}),sc=_t(Va),A_=we({},Va,{dataTransfer:0}),L_=_t(A_),j_=we({},oo,{relatedTarget:0}),ks=_t(j_),R_=we({},$r,{animationName:0,elapsedTime:0,pseudoElement:0}),F_=_t(R_),I_=we({},$r,{clipboardData:function(e){return"clipboardData"in e?e.clipboardData:window.clipboardData}}),D_=_t(I_),M_=we({},$r,{data:0}),lc=_t(M_),q_={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},O_={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},W_={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function V_(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=W_[e])?!!t[e]:!1}function Bu(){return V_}var B_=we({},oo,{key:function(e){if(e.key){var t=q_[e.key]||e.key;if(t!=="Unidentified")return t}return e.type==="keypress"?(e=Ko(e),e===13?"Enter":String.fromCharCode(e)):e.type==="keydown"||e.type==="keyup"?O_[e.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:Bu,charCode:function(e){return e.type==="keypress"?Ko(e):0},keyCode:function(e){return e.type==="keydown"||e.type==="keyup"?e.keyCode:0},which:function(e){return e.type==="keypress"?Ko(e):e.type==="keydown"||e.type==="keyup"?e.keyCode:0}}),U_=_t(B_),H_=we({},Va,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),uc=_t(H_),Q_=we({},oo,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:Bu}),X_=_t(Q_),K_=we({},$r,{propertyName:0,elapsedTime:0,pseudoElement:0}),Y_=_t(K_),$_=we({},Va,{deltaX:function(e){return"deltaX"in e?e.deltaX:"wheelDeltaX"in e?-e.wheelDeltaX:0},deltaY:function(e){return"deltaY"in e?e.deltaY:"wheelDeltaY"in e?-e.wheelDeltaY:"wheelDelta"in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0}),G_=_t($_),Z_=[9,13,27,32],Uu=on&&"CompositionEvent"in window,wi=null;on&&"documentMode"in document&&(wi=document.documentMode);var J_=on&&"TextEvent"in window&&!wi,xm=on&&(!Uu||wi&&8<wi&&11>=wi),dc=" ",cc=!1;function bm(e,t){switch(e){case"keyup":return Z_.indexOf(t.keyCode)!==-1;case"keydown":return t.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function vm(e){return e=e.detail,typeof e=="object"&&"data"in e?e.data:null}var xr=!1;function ey(e,t){switch(e){case"compositionend":return vm(t);case"keypress":return t.which!==32?null:(cc=!0,dc);case"textInput":return e=t.data,e===dc&&cc?null:e;default:return null}}function ty(e,t){if(xr)return e==="compositionend"||!Uu&&bm(e,t)?(e=ym(),Xo=Wu=kn=null,xr=!1,e):null;switch(e){case"paste":return null;case"keypress":if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case"compositionend":return xm&&t.locale!=="ko"?null:t.data;default:return null}}var ny={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function pc(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t==="input"?!!ny[e.type]:t==="textarea"}function km(e,t,n,r){Jh(r),t=ga(t,"onChange"),0<t.length&&(n=new Vu("onChange","change",null,n,r),e.push({event:n,listeners:t}))}var Ti=null,Oi=null;function ry(e){jm(e,0)}function Ba(e){var t=kr(e);if(Qh(t))return e}function iy(e,t){if(e==="change")return t}var wm=!1;if(on){var ws;if(on){var Ts="oninput"in document;if(!Ts){var hc=document.createElement("div");hc.setAttribute("oninput","return;"),Ts=typeof hc.oninput=="function"}ws=Ts}else ws=!1;wm=ws&&(!document.documentMode||9<document.documentMode)}function mc(){Ti&&(Ti.detachEvent("onpropertychange",Tm),Oi=Ti=null)}function Tm(e){if(e.propertyName==="value"&&Ba(Oi)){var t=[];km(t,Oi,e,Iu(e)),rm(ry,t)}}function oy(e,t,n){e==="focusin"?(mc(),Ti=t,Oi=n,Ti.attachEvent("onpropertychange",Tm)):e==="focusout"&&mc()}function ay(e){if(e==="selectionchange"||e==="keyup"||e==="keydown")return Ba(Oi)}function sy(e,t){if(e==="click")return Ba(t)}function ly(e,t){if(e==="input"||e==="change")return Ba(t)}function uy(e,t){return e===t&&(e!==0||1/e===1/t)||e!==e&&t!==t}var Wt=typeof Object.is=="function"?Object.is:uy;function Wi(e,t){if(Wt(e,t))return!0;if(typeof e!="object"||e===null||typeof t!="object"||t===null)return!1;var n=Object.keys(e),r=Object.keys(t);if(n.length!==r.length)return!1;for(r=0;r<n.length;r++){var i=n[r];if(!hl.call(t,i)||!Wt(e[i],t[i]))return!1}return!0}function fc(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function gc(e,t){var n=fc(e);e=0;for(var r;n;){if(n.nodeType===3){if(r=e+n.textContent.length,e<=t&&r>=t)return{node:n,offset:t-e};e=r}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=fc(n)}}function Cm(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?Cm(e,t.parentNode):"contains"in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function Sm(){for(var e=window,t=ua();t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href=="string"}catch{n=!1}if(n)e=t.contentWindow;else break;t=ua(e.document)}return t}function Hu(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t==="input"&&(e.type==="text"||e.type==="search"||e.type==="tel"||e.type==="url"||e.type==="password")||t==="textarea"||e.contentEditable==="true")}function dy(e){var t=Sm(),n=e.focusedElem,r=e.selectionRange;if(t!==n&&n&&n.ownerDocument&&Cm(n.ownerDocument.documentElement,n)){if(r!==null&&Hu(n)){if(t=r.start,e=r.end,e===void 0&&(e=t),"selectionStart"in n)n.selectionStart=t,n.selectionEnd=Math.min(e,n.value.length);else if(e=(t=n.ownerDocument||document)&&t.defaultView||window,e.getSelection){e=e.getSelection();var i=n.textContent.length,o=Math.min(r.start,i);r=r.end===void 0?o:Math.min(r.end,i),!e.extend&&o>r&&(i=r,r=o,o=i),i=gc(n,o);var a=gc(n,r);i&&a&&(e.rangeCount!==1||e.anchorNode!==i.node||e.anchorOffset!==i.offset||e.focusNode!==a.node||e.focusOffset!==a.offset)&&(t=t.createRange(),t.setStart(i.node,i.offset),e.removeAllRanges(),o>r?(e.addRange(t),e.extend(a.node,a.offset)):(t.setEnd(a.node,a.offset),e.addRange(t)))}}for(t=[],e=n;e=e.parentNode;)e.nodeType===1&&t.push({element:e,left:e.scrollLeft,top:e.scrollTop});for(typeof n.focus=="function"&&n.focus(),n=0;n<t.length;n++)e=t[n],e.element.scrollLeft=e.left,e.element.scrollTop=e.top}}var cy=on&&"documentMode"in document&&11>=document.documentMode,br=null,Ll=null,Ci=null,jl=!1;function _c(e,t,n){var r=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;jl||br==null||br!==ua(r)||(r=br,"selectionStart"in r&&Hu(r)?r={start:r.selectionStart,end:r.selectionEnd}:(r=(r.ownerDocument&&r.ownerDocument.defaultView||window).getSelection(),r={anchorNode:r.anchorNode,anchorOffset:r.anchorOffset,focusNode:r.focusNode,focusOffset:r.focusOffset}),Ci&&Wi(Ci,r)||(Ci=r,r=ga(Ll,"onSelect"),0<r.length&&(t=new Vu("onSelect","select",null,t,n),e.push({event:t,listeners:r}),t.target=br)))}function So(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n["Webkit"+e]="webkit"+t,n["Moz"+e]="moz"+t,n}var vr={animationend:So("Animation","AnimationEnd"),animationiteration:So("Animation","AnimationIteration"),animationstart:So("Animation","AnimationStart"),transitionend:So("Transition","TransitionEnd")},Cs={},zm={};on&&(zm=document.createElement("div").style,"AnimationEvent"in window||(delete vr.animationend.animation,delete vr.animationiteration.animation,delete vr.animationstart.animation),"TransitionEvent"in window||delete vr.transitionend.transition);function Ua(e){if(Cs[e])return Cs[e];if(!vr[e])return e;var t=vr[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in zm)return Cs[e]=t[n];return e}var Em=Ua("animationend"),Pm=Ua("animationiteration"),Nm=Ua("animationstart"),Am=Ua("transitionend"),Lm=new Map,yc="abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");function Mn(e,t){Lm.set(e,t),sr(t,[e])}for(var Ss=0;Ss<yc.length;Ss++){var zs=yc[Ss],py=zs.toLowerCase(),hy=zs[0].toUpperCase()+zs.slice(1);Mn(py,"on"+hy)}Mn(Em,"onAnimationEnd");Mn(Pm,"onAnimationIteration");Mn(Nm,"onAnimationStart");Mn("dblclick","onDoubleClick");Mn("focusin","onFocus");Mn("focusout","onBlur");Mn(Am,"onTransitionEnd");Or("onMouseEnter",["mouseout","mouseover"]);Or("onMouseLeave",["mouseout","mouseover"]);Or("onPointerEnter",["pointerout","pointerover"]);Or("onPointerLeave",["pointerout","pointerover"]);sr("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));sr("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));sr("onBeforeInput",["compositionend","keypress","textInput","paste"]);sr("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));sr("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));sr("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var xi="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),my=new Set("cancel close invalid load scroll toggle".split(" ").concat(xi));function xc(e,t,n){var r=e.type||"unknown-event";e.currentTarget=n,p_(r,t,void 0,e),e.currentTarget=null}function jm(e,t){t=(t&4)!==0;for(var n=0;n<e.length;n++){var r=e[n],i=r.event;r=r.listeners;e:{var o=void 0;if(t)for(var a=r.length-1;0<=a;a--){var s=r[a],l=s.instance,u=s.currentTarget;if(s=s.listener,l!==o&&i.isPropagationStopped())break e;xc(i,s,u),o=l}else for(a=0;a<r.length;a++){if(s=r[a],l=s.instance,u=s.currentTarget,s=s.listener,l!==o&&i.isPropagationStopped())break e;xc(i,s,u),o=l}}}if(ca)throw e=El,ca=!1,El=null,e}function ge(e,t){var n=t[Ml];n===void 0&&(n=t[Ml]=new Set);var r=e+"__bubble";n.has(r)||(Rm(t,e,2,!1),n.add(r))}function Es(e,t,n){var r=0;t&&(r|=4),Rm(n,e,r,t)}var zo="_reactListening"+Math.random().toString(36).slice(2);function Vi(e){if(!e[zo]){e[zo]=!0,Wh.forEach(function(n){n!=="selectionchange"&&(my.has(n)||Es(n,!1,e),Es(n,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[zo]||(t[zo]=!0,Es("selectionchange",!1,t))}}function Rm(e,t,n,r){switch(_m(t)){case 1:var i=E_;break;case 4:i=P_;break;default:i=Ou}n=i.bind(null,t,n,e),i=void 0,!zl||t!=="touchstart"&&t!=="touchmove"&&t!=="wheel"||(i=!0),r?i!==void 0?e.addEventListener(t,n,{capture:!0,passive:i}):e.addEventListener(t,n,!0):i!==void 0?e.addEventListener(t,n,{passive:i}):e.addEventListener(t,n,!1)}function Ps(e,t,n,r,i){var o=r;if(!(t&1)&&!(t&2)&&r!==null)e:for(;;){if(r===null)return;var a=r.tag;if(a===3||a===4){var s=r.stateNode.containerInfo;if(s===i||s.nodeType===8&&s.parentNode===i)break;if(a===4)for(a=r.return;a!==null;){var l=a.tag;if((l===3||l===4)&&(l=a.stateNode.containerInfo,l===i||l.nodeType===8&&l.parentNode===i))return;a=a.return}for(;s!==null;){if(a=Kn(s),a===null)return;if(l=a.tag,l===5||l===6){r=o=a;continue e}s=s.parentNode}}r=r.return}rm(function(){var u=o,c=Iu(n),d=[];e:{var h=Lm.get(e);if(h!==void 0){var m=Vu,f=e;switch(e){case"keypress":if(Ko(n)===0)break e;case"keydown":case"keyup":m=U_;break;case"focusin":f="focus",m=ks;break;case"focusout":f="blur",m=ks;break;case"beforeblur":case"afterblur":m=ks;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":m=sc;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":m=L_;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":m=X_;break;case Em:case Pm:case Nm:m=F_;break;case Am:m=Y_;break;case"scroll":m=N_;break;case"wheel":m=G_;break;case"copy":case"cut":case"paste":m=D_;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":m=uc}var x=(t&4)!==0,S=!x&&e==="scroll",g=x?h!==null?h+"Capture":null:h;x=[];for(var y=u,_;y!==null;){_=y;var w=_.stateNode;if(_.tag===5&&w!==null&&(_=w,g!==null&&(w=Ii(y,g),w!=null&&x.push(Bi(y,w,_)))),S)break;y=y.return}0<x.length&&(h=new m(h,f,null,n,c),d.push({event:h,listeners:x}))}}if(!(t&7)){e:{if(h=e==="mouseover"||e==="pointerover",m=e==="mouseout"||e==="pointerout",h&&n!==Cl&&(f=n.relatedTarget||n.fromElement)&&(Kn(f)||f[an]))break e;if((m||h)&&(h=c.window===c?c:(h=c.ownerDocument)?h.defaultView||h.parentWindow:window,m?(f=n.relatedTarget||n.toElement,m=u,f=f?Kn(f):null,f!==null&&(S=lr(f),f!==S||f.tag!==5&&f.tag!==6)&&(f=null)):(m=null,f=u),m!==f)){if(x=sc,w="onMouseLeave",g="onMouseEnter",y="mouse",(e==="pointerout"||e==="pointerover")&&(x=uc,w="onPointerLeave",g="onPointerEnter",y="pointer"),S=m==null?h:kr(m),_=f==null?h:kr(f),h=new x(w,y+"leave",m,n,c),h.target=S,h.relatedTarget=_,w=null,Kn(c)===u&&(x=new x(g,y+"enter",f,n,c),x.target=_,x.relatedTarget=S,w=x),S=w,m&&f)t:{for(x=m,g=f,y=0,_=x;_;_=hr(_))y++;for(_=0,w=g;w;w=hr(w))_++;for(;0<y-_;)x=hr(x),y--;for(;0<_-y;)g=hr(g),_--;for(;y--;){if(x===g||g!==null&&x===g.alternate)break t;x=hr(x),g=hr(g)}x=null}else x=null;m!==null&&bc(d,h,m,x,!1),f!==null&&S!==null&&bc(d,S,f,x,!0)}}e:{if(h=u?kr(u):window,m=h.nodeName&&h.nodeName.toLowerCase(),m==="select"||m==="input"&&h.type==="file")var z=iy;else if(pc(h))if(wm)z=ly;else{z=ay;var k=oy}else(m=h.nodeName)&&m.toLowerCase()==="input"&&(h.type==="checkbox"||h.type==="radio")&&(z=sy);if(z&&(z=z(e,u))){km(d,z,n,c);break e}k&&k(e,h,u),e==="focusout"&&(k=h._wrapperState)&&k.controlled&&h.type==="number"&&bl(h,"number",h.value)}switch(k=u?kr(u):window,e){case"focusin":(pc(k)||k.contentEditable==="true")&&(br=k,Ll=u,Ci=null);break;case"focusout":Ci=Ll=br=null;break;case"mousedown":jl=!0;break;case"contextmenu":case"mouseup":case"dragend":jl=!1,_c(d,n,c);break;case"selectionchange":if(cy)break;case"keydown":case"keyup":_c(d,n,c)}var E;if(Uu)e:{switch(e){case"compositionstart":var N="onCompositionStart";break e;case"compositionend":N="onCompositionEnd";break e;case"compositionupdate":N="onCompositionUpdate";break e}N=void 0}else xr?bm(e,n)&&(N="onCompositionEnd"):e==="keydown"&&n.keyCode===229&&(N="onCompositionStart");N&&(xm&&n.locale!=="ko"&&(xr||N!=="onCompositionStart"?N==="onCompositionEnd"&&xr&&(E=ym()):(kn=c,Wu="value"in kn?kn.value:kn.textContent,xr=!0)),k=ga(u,N),0<k.length&&(N=new lc(N,e,null,n,c),d.push({event:N,listeners:k}),E?N.data=E:(E=vm(n),E!==null&&(N.data=E)))),(E=J_?ey(e,n):ty(e,n))&&(u=ga(u,"onBeforeInput"),0<u.length&&(c=new lc("onBeforeInput","beforeinput",null,n,c),d.push({event:c,listeners:u}),c.data=E))}jm(d,t)})}function Bi(e,t,n){return{instance:e,listener:t,currentTarget:n}}function ga(e,t){for(var n=t+"Capture",r=[];e!==null;){var i=e,o=i.stateNode;i.tag===5&&o!==null&&(i=o,o=Ii(e,n),o!=null&&r.unshift(Bi(e,o,i)),o=Ii(e,t),o!=null&&r.push(Bi(e,o,i))),e=e.return}return r}function hr(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5);return e||null}function bc(e,t,n,r,i){for(var o=t._reactName,a=[];n!==null&&n!==r;){var s=n,l=s.alternate,u=s.stateNode;if(l!==null&&l===r)break;s.tag===5&&u!==null&&(s=u,i?(l=Ii(n,o),l!=null&&a.unshift(Bi(n,l,s))):i||(l=Ii(n,o),l!=null&&a.push(Bi(n,l,s)))),n=n.return}a.length!==0&&e.push({event:t,listeners:a})}var fy=/\r\n?/g,gy=/\u0000|\uFFFD/g;function vc(e){return(typeof e=="string"?e:""+e).replace(fy,`
`).replace(gy,"")}function Eo(e,t,n){if(t=vc(t),vc(e)!==t&&n)throw Error(L(425))}function _a(){}var Rl=null,Fl=null;function Il(e,t){return e==="textarea"||e==="noscript"||typeof t.children=="string"||typeof t.children=="number"||typeof t.dangerouslySetInnerHTML=="object"&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var Dl=typeof setTimeout=="function"?setTimeout:void 0,_y=typeof clearTimeout=="function"?clearTimeout:void 0,kc=typeof Promise=="function"?Promise:void 0,yy=typeof queueMicrotask=="function"?queueMicrotask:typeof kc<"u"?function(e){return kc.resolve(null).then(e).catch(xy)}:Dl;function xy(e){setTimeout(function(){throw e})}function Ns(e,t){var n=t,r=0;do{var i=n.nextSibling;if(e.removeChild(n),i&&i.nodeType===8)if(n=i.data,n==="/$"){if(r===0){e.removeChild(i),qi(t);return}r--}else n!=="$"&&n!=="$?"&&n!=="$!"||r++;n=i}while(n);qi(t)}function Pn(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t==="$"||t==="$!"||t==="$?")break;if(t==="/$")return null}}return e}function wc(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="$"||n==="$!"||n==="$?"){if(t===0)return e;t--}else n==="/$"&&t++}e=e.previousSibling}return null}var Gr=Math.random().toString(36).slice(2),Qt="__reactFiber$"+Gr,Ui="__reactProps$"+Gr,an="__reactContainer$"+Gr,Ml="__reactEvents$"+Gr,by="__reactListeners$"+Gr,vy="__reactHandles$"+Gr;function Kn(e){var t=e[Qt];if(t)return t;for(var n=e.parentNode;n;){if(t=n[an]||n[Qt]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=wc(e);e!==null;){if(n=e[Qt])return n;e=wc(e)}return t}e=n,n=e.parentNode}return null}function ao(e){return e=e[Qt]||e[an],!e||e.tag!==5&&e.tag!==6&&e.tag!==13&&e.tag!==3?null:e}function kr(e){if(e.tag===5||e.tag===6)return e.stateNode;throw Error(L(33))}function Ha(e){return e[Ui]||null}var ql=[],wr=-1;function qn(e){return{current:e}}function _e(e){0>wr||(e.current=ql[wr],ql[wr]=null,wr--)}function me(e,t){wr++,ql[wr]=e.current,e.current=t}var Dn={},He=qn(Dn),ot=qn(!1),er=Dn;function Wr(e,t){var n=e.type.contextTypes;if(!n)return Dn;var r=e.stateNode;if(r&&r.__reactInternalMemoizedUnmaskedChildContext===t)return r.__reactInternalMemoizedMaskedChildContext;var i={},o;for(o in n)i[o]=t[o];return r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=t,e.__reactInternalMemoizedMaskedChildContext=i),i}function at(e){return e=e.childContextTypes,e!=null}function ya(){_e(ot),_e(He)}function Tc(e,t,n){if(He.current!==Dn)throw Error(L(168));me(He,t),me(ot,n)}function Fm(e,t,n){var r=e.stateNode;if(t=t.childContextTypes,typeof r.getChildContext!="function")return n;r=r.getChildContext();for(var i in r)if(!(i in t))throw Error(L(108,o_(e)||"Unknown",i));return we({},n,r)}function xa(e){return e=(e=e.stateNode)&&e.__reactInternalMemoizedMergedChildContext||Dn,er=He.current,me(He,e),me(ot,ot.current),!0}function Cc(e,t,n){var r=e.stateNode;if(!r)throw Error(L(169));n?(e=Fm(e,t,er),r.__reactInternalMemoizedMergedChildContext=e,_e(ot),_e(He),me(He,e)):_e(ot),me(ot,n)}var en=null,Qa=!1,As=!1;function Im(e){en===null?en=[e]:en.push(e)}function ky(e){Qa=!0,Im(e)}function On(){if(!As&&en!==null){As=!0;var e=0,t=de;try{var n=en;for(de=1;e<n.length;e++){var r=n[e];do r=r(!0);while(r!==null)}en=null,Qa=!1}catch(i){throw en!==null&&(en=en.slice(e+1)),sm(Du,On),i}finally{de=t,As=!1}}return null}var Tr=[],Cr=0,ba=null,va=0,vt=[],kt=0,tr=null,tn=1,nn="";function Un(e,t){Tr[Cr++]=va,Tr[Cr++]=ba,ba=e,va=t}function Dm(e,t,n){vt[kt++]=tn,vt[kt++]=nn,vt[kt++]=tr,tr=e;var r=tn;e=nn;var i=32-Mt(r)-1;r&=~(1<<i),n+=1;var o=32-Mt(t)+i;if(30<o){var a=i-i%5;o=(r&(1<<a)-1).toString(32),r>>=a,i-=a,tn=1<<32-Mt(t)+i|n<<i|r,nn=o+e}else tn=1<<o|n<<i|r,nn=e}function Qu(e){e.return!==null&&(Un(e,1),Dm(e,1,0))}function Xu(e){for(;e===ba;)ba=Tr[--Cr],Tr[Cr]=null,va=Tr[--Cr],Tr[Cr]=null;for(;e===tr;)tr=vt[--kt],vt[kt]=null,nn=vt[--kt],vt[kt]=null,tn=vt[--kt],vt[kt]=null}var ht=null,pt=null,ye=!1,It=null;function Mm(e,t){var n=Tt(5,null,null,0);n.elementType="DELETED",n.stateNode=t,n.return=e,t=e.deletions,t===null?(e.deletions=[n],e.flags|=16):t.push(n)}function Sc(e,t){switch(e.tag){case 5:var n=e.type;return t=t.nodeType!==1||n.toLowerCase()!==t.nodeName.toLowerCase()?null:t,t!==null?(e.stateNode=t,ht=e,pt=Pn(t.firstChild),!0):!1;case 6:return t=e.pendingProps===""||t.nodeType!==3?null:t,t!==null?(e.stateNode=t,ht=e,pt=null,!0):!1;case 13:return t=t.nodeType!==8?null:t,t!==null?(n=tr!==null?{id:tn,overflow:nn}:null,e.memoizedState={dehydrated:t,treeContext:n,retryLane:1073741824},n=Tt(18,null,null,0),n.stateNode=t,n.return=e,e.child=n,ht=e,pt=null,!0):!1;default:return!1}}function Ol(e){return(e.mode&1)!==0&&(e.flags&128)===0}function Wl(e){if(ye){var t=pt;if(t){var n=t;if(!Sc(e,t)){if(Ol(e))throw Error(L(418));t=Pn(n.nextSibling);var r=ht;t&&Sc(e,t)?Mm(r,n):(e.flags=e.flags&-4097|2,ye=!1,ht=e)}}else{if(Ol(e))throw Error(L(418));e.flags=e.flags&-4097|2,ye=!1,ht=e}}}function zc(e){for(e=e.return;e!==null&&e.tag!==5&&e.tag!==3&&e.tag!==13;)e=e.return;ht=e}function Po(e){if(e!==ht)return!1;if(!ye)return zc(e),ye=!0,!1;var t;if((t=e.tag!==3)&&!(t=e.tag!==5)&&(t=e.type,t=t!=="head"&&t!=="body"&&!Il(e.type,e.memoizedProps)),t&&(t=pt)){if(Ol(e))throw qm(),Error(L(418));for(;t;)Mm(e,t),t=Pn(t.nextSibling)}if(zc(e),e.tag===13){if(e=e.memoizedState,e=e!==null?e.dehydrated:null,!e)throw Error(L(317));e:{for(e=e.nextSibling,t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="/$"){if(t===0){pt=Pn(e.nextSibling);break e}t--}else n!=="$"&&n!=="$!"&&n!=="$?"||t++}e=e.nextSibling}pt=null}}else pt=ht?Pn(e.stateNode.nextSibling):null;return!0}function qm(){for(var e=pt;e;)e=Pn(e.nextSibling)}function Vr(){pt=ht=null,ye=!1}function Ku(e){It===null?It=[e]:It.push(e)}var wy=un.ReactCurrentBatchConfig;function li(e,t,n){if(e=n.ref,e!==null&&typeof e!="function"&&typeof e!="object"){if(n._owner){if(n=n._owner,n){if(n.tag!==1)throw Error(L(309));var r=n.stateNode}if(!r)throw Error(L(147,e));var i=r,o=""+e;return t!==null&&t.ref!==null&&typeof t.ref=="function"&&t.ref._stringRef===o?t.ref:(t=function(a){var s=i.refs;a===null?delete s[o]:s[o]=a},t._stringRef=o,t)}if(typeof e!="string")throw Error(L(284));if(!n._owner)throw Error(L(290,e))}return e}function No(e,t){throw e=Object.prototype.toString.call(t),Error(L(31,e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e))}function Ec(e){var t=e._init;return t(e._payload)}function Om(e){function t(g,y){if(e){var _=g.deletions;_===null?(g.deletions=[y],g.flags|=16):_.push(y)}}function n(g,y){if(!e)return null;for(;y!==null;)t(g,y),y=y.sibling;return null}function r(g,y){for(g=new Map;y!==null;)y.key!==null?g.set(y.key,y):g.set(y.index,y),y=y.sibling;return g}function i(g,y){return g=jn(g,y),g.index=0,g.sibling=null,g}function o(g,y,_){return g.index=_,e?(_=g.alternate,_!==null?(_=_.index,_<y?(g.flags|=2,y):_):(g.flags|=2,y)):(g.flags|=1048576,y)}function a(g){return e&&g.alternate===null&&(g.flags|=2),g}function s(g,y,_,w){return y===null||y.tag!==6?(y=Ms(_,g.mode,w),y.return=g,y):(y=i(y,_),y.return=g,y)}function l(g,y,_,w){var z=_.type;return z===yr?c(g,y,_.props.children,w,_.key):y!==null&&(y.elementType===z||typeof z=="object"&&z!==null&&z.$$typeof===yn&&Ec(z)===y.type)?(w=i(y,_.props),w.ref=li(g,y,_),w.return=g,w):(w=ta(_.type,_.key,_.props,null,g.mode,w),w.ref=li(g,y,_),w.return=g,w)}function u(g,y,_,w){return y===null||y.tag!==4||y.stateNode.containerInfo!==_.containerInfo||y.stateNode.implementation!==_.implementation?(y=qs(_,g.mode,w),y.return=g,y):(y=i(y,_.children||[]),y.return=g,y)}function c(g,y,_,w,z){return y===null||y.tag!==7?(y=Jn(_,g.mode,w,z),y.return=g,y):(y=i(y,_),y.return=g,y)}function d(g,y,_){if(typeof y=="string"&&y!==""||typeof y=="number")return y=Ms(""+y,g.mode,_),y.return=g,y;if(typeof y=="object"&&y!==null){switch(y.$$typeof){case xo:return _=ta(y.type,y.key,y.props,null,g.mode,_),_.ref=li(g,null,y),_.return=g,_;case _r:return y=qs(y,g.mode,_),y.return=g,y;case yn:var w=y._init;return d(g,w(y._payload),_)}if(_i(y)||ri(y))return y=Jn(y,g.mode,_,null),y.return=g,y;No(g,y)}return null}function h(g,y,_,w){var z=y!==null?y.key:null;if(typeof _=="string"&&_!==""||typeof _=="number")return z!==null?null:s(g,y,""+_,w);if(typeof _=="object"&&_!==null){switch(_.$$typeof){case xo:return _.key===z?l(g,y,_,w):null;case _r:return _.key===z?u(g,y,_,w):null;case yn:return z=_._init,h(g,y,z(_._payload),w)}if(_i(_)||ri(_))return z!==null?null:c(g,y,_,w,null);No(g,_)}return null}function m(g,y,_,w,z){if(typeof w=="string"&&w!==""||typeof w=="number")return g=g.get(_)||null,s(y,g,""+w,z);if(typeof w=="object"&&w!==null){switch(w.$$typeof){case xo:return g=g.get(w.key===null?_:w.key)||null,l(y,g,w,z);case _r:return g=g.get(w.key===null?_:w.key)||null,u(y,g,w,z);case yn:var k=w._init;return m(g,y,_,k(w._payload),z)}if(_i(w)||ri(w))return g=g.get(_)||null,c(y,g,w,z,null);No(y,w)}return null}function f(g,y,_,w){for(var z=null,k=null,E=y,N=y=0,q=null;E!==null&&N<_.length;N++){E.index>N?(q=E,E=null):q=E.sibling;var T=h(g,E,_[N],w);if(T===null){E===null&&(E=q);break}e&&E&&T.alternate===null&&t(g,E),y=o(T,y,N),k===null?z=T:k.sibling=T,k=T,E=q}if(N===_.length)return n(g,E),ye&&Un(g,N),z;if(E===null){for(;N<_.length;N++)E=d(g,_[N],w),E!==null&&(y=o(E,y,N),k===null?z=E:k.sibling=E,k=E);return ye&&Un(g,N),z}for(E=r(g,E);N<_.length;N++)q=m(E,g,N,_[N],w),q!==null&&(e&&q.alternate!==null&&E.delete(q.key===null?N:q.key),y=o(q,y,N),k===null?z=q:k.sibling=q,k=q);return e&&E.forEach(function(D){return t(g,D)}),ye&&Un(g,N),z}function x(g,y,_,w){var z=ri(_);if(typeof z!="function")throw Error(L(150));if(_=z.call(_),_==null)throw Error(L(151));for(var k=z=null,E=y,N=y=0,q=null,T=_.next();E!==null&&!T.done;N++,T=_.next()){E.index>N?(q=E,E=null):q=E.sibling;var D=h(g,E,T.value,w);if(D===null){E===null&&(E=q);break}e&&E&&D.alternate===null&&t(g,E),y=o(D,y,N),k===null?z=D:k.sibling=D,k=D,E=q}if(T.done)return n(g,E),ye&&Un(g,N),z;if(E===null){for(;!T.done;N++,T=_.next())T=d(g,T.value,w),T!==null&&(y=o(T,y,N),k===null?z=T:k.sibling=T,k=T);return ye&&Un(g,N),z}for(E=r(g,E);!T.done;N++,T=_.next())T=m(E,g,N,T.value,w),T!==null&&(e&&T.alternate!==null&&E.delete(T.key===null?N:T.key),y=o(T,y,N),k===null?z=T:k.sibling=T,k=T);return e&&E.forEach(function(W){return t(g,W)}),ye&&Un(g,N),z}function S(g,y,_,w){if(typeof _=="object"&&_!==null&&_.type===yr&&_.key===null&&(_=_.props.children),typeof _=="object"&&_!==null){switch(_.$$typeof){case xo:e:{for(var z=_.key,k=y;k!==null;){if(k.key===z){if(z=_.type,z===yr){if(k.tag===7){n(g,k.sibling),y=i(k,_.props.children),y.return=g,g=y;break e}}else if(k.elementType===z||typeof z=="object"&&z!==null&&z.$$typeof===yn&&Ec(z)===k.type){n(g,k.sibling),y=i(k,_.props),y.ref=li(g,k,_),y.return=g,g=y;break e}n(g,k);break}else t(g,k);k=k.sibling}_.type===yr?(y=Jn(_.props.children,g.mode,w,_.key),y.return=g,g=y):(w=ta(_.type,_.key,_.props,null,g.mode,w),w.ref=li(g,y,_),w.return=g,g=w)}return a(g);case _r:e:{for(k=_.key;y!==null;){if(y.key===k)if(y.tag===4&&y.stateNode.containerInfo===_.containerInfo&&y.stateNode.implementation===_.implementation){n(g,y.sibling),y=i(y,_.children||[]),y.return=g,g=y;break e}else{n(g,y);break}else t(g,y);y=y.sibling}y=qs(_,g.mode,w),y.return=g,g=y}return a(g);case yn:return k=_._init,S(g,y,k(_._payload),w)}if(_i(_))return f(g,y,_,w);if(ri(_))return x(g,y,_,w);No(g,_)}return typeof _=="string"&&_!==""||typeof _=="number"?(_=""+_,y!==null&&y.tag===6?(n(g,y.sibling),y=i(y,_),y.return=g,g=y):(n(g,y),y=Ms(_,g.mode,w),y.return=g,g=y),a(g)):n(g,y)}return S}var Br=Om(!0),Wm=Om(!1),ka=qn(null),wa=null,Sr=null,Yu=null;function $u(){Yu=Sr=wa=null}function Gu(e){var t=ka.current;_e(ka),e._currentValue=t}function Vl(e,t,n){for(;e!==null;){var r=e.alternate;if((e.childLanes&t)!==t?(e.childLanes|=t,r!==null&&(r.childLanes|=t)):r!==null&&(r.childLanes&t)!==t&&(r.childLanes|=t),e===n)break;e=e.return}}function Rr(e,t){wa=e,Yu=Sr=null,e=e.dependencies,e!==null&&e.firstContext!==null&&(e.lanes&t&&(it=!0),e.firstContext=null)}function St(e){var t=e._currentValue;if(Yu!==e)if(e={context:e,memoizedValue:t,next:null},Sr===null){if(wa===null)throw Error(L(308));Sr=e,wa.dependencies={lanes:0,firstContext:e}}else Sr=Sr.next=e;return t}var Yn=null;function Zu(e){Yn===null?Yn=[e]:Yn.push(e)}function Vm(e,t,n,r){var i=t.interleaved;return i===null?(n.next=n,Zu(t)):(n.next=i.next,i.next=n),t.interleaved=n,sn(e,r)}function sn(e,t){e.lanes|=t;var n=e.alternate;for(n!==null&&(n.lanes|=t),n=e,e=e.return;e!==null;)e.childLanes|=t,n=e.alternate,n!==null&&(n.childLanes|=t),n=e,e=e.return;return n.tag===3?n.stateNode:null}var xn=!1;function Ju(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,interleaved:null,lanes:0},effects:null}}function Bm(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,effects:e.effects})}function rn(e,t){return{eventTime:e,lane:t,tag:0,payload:null,callback:null,next:null}}function Nn(e,t,n){var r=e.updateQueue;if(r===null)return null;if(r=r.shared,ae&2){var i=r.pending;return i===null?t.next=t:(t.next=i.next,i.next=t),r.pending=t,sn(e,n)}return i=r.interleaved,i===null?(t.next=t,Zu(r)):(t.next=i.next,i.next=t),r.interleaved=t,sn(e,n)}function Yo(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,(n&4194240)!==0)){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,Mu(e,n)}}function Pc(e,t){var n=e.updateQueue,r=e.alternate;if(r!==null&&(r=r.updateQueue,n===r)){var i=null,o=null;if(n=n.firstBaseUpdate,n!==null){do{var a={eventTime:n.eventTime,lane:n.lane,tag:n.tag,payload:n.payload,callback:n.callback,next:null};o===null?i=o=a:o=o.next=a,n=n.next}while(n!==null);o===null?i=o=t:o=o.next=t}else i=o=t;n={baseState:r.baseState,firstBaseUpdate:i,lastBaseUpdate:o,shared:r.shared,effects:r.effects},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}function Ta(e,t,n,r){var i=e.updateQueue;xn=!1;var o=i.firstBaseUpdate,a=i.lastBaseUpdate,s=i.shared.pending;if(s!==null){i.shared.pending=null;var l=s,u=l.next;l.next=null,a===null?o=u:a.next=u,a=l;var c=e.alternate;c!==null&&(c=c.updateQueue,s=c.lastBaseUpdate,s!==a&&(s===null?c.firstBaseUpdate=u:s.next=u,c.lastBaseUpdate=l))}if(o!==null){var d=i.baseState;a=0,c=u=l=null,s=o;do{var h=s.lane,m=s.eventTime;if((r&h)===h){c!==null&&(c=c.next={eventTime:m,lane:0,tag:s.tag,payload:s.payload,callback:s.callback,next:null});e:{var f=e,x=s;switch(h=t,m=n,x.tag){case 1:if(f=x.payload,typeof f=="function"){d=f.call(m,d,h);break e}d=f;break e;case 3:f.flags=f.flags&-65537|128;case 0:if(f=x.payload,h=typeof f=="function"?f.call(m,d,h):f,h==null)break e;d=we({},d,h);break e;case 2:xn=!0}}s.callback!==null&&s.lane!==0&&(e.flags|=64,h=i.effects,h===null?i.effects=[s]:h.push(s))}else m={eventTime:m,lane:h,tag:s.tag,payload:s.payload,callback:s.callback,next:null},c===null?(u=c=m,l=d):c=c.next=m,a|=h;if(s=s.next,s===null){if(s=i.shared.pending,s===null)break;h=s,s=h.next,h.next=null,i.lastBaseUpdate=h,i.shared.pending=null}}while(!0);if(c===null&&(l=d),i.baseState=l,i.firstBaseUpdate=u,i.lastBaseUpdate=c,t=i.shared.interleaved,t!==null){i=t;do a|=i.lane,i=i.next;while(i!==t)}else o===null&&(i.shared.lanes=0);rr|=a,e.lanes=a,e.memoizedState=d}}function Nc(e,t,n){if(e=t.effects,t.effects=null,e!==null)for(t=0;t<e.length;t++){var r=e[t],i=r.callback;if(i!==null){if(r.callback=null,r=n,typeof i!="function")throw Error(L(191,i));i.call(r)}}}var so={},Kt=qn(so),Hi=qn(so),Qi=qn(so);function $n(e){if(e===so)throw Error(L(174));return e}function ed(e,t){switch(me(Qi,t),me(Hi,e),me(Kt,so),e=t.nodeType,e){case 9:case 11:t=(t=t.documentElement)?t.namespaceURI:kl(null,"");break;default:e=e===8?t.parentNode:t,t=e.namespaceURI||null,e=e.tagName,t=kl(t,e)}_e(Kt),me(Kt,t)}function Ur(){_e(Kt),_e(Hi),_e(Qi)}function Um(e){$n(Qi.current);var t=$n(Kt.current),n=kl(t,e.type);t!==n&&(me(Hi,e),me(Kt,n))}function td(e){Hi.current===e&&(_e(Kt),_e(Hi))}var be=qn(0);function Ca(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||n.data==="$?"||n.data==="$!"))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==void 0){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var Ls=[];function nd(){for(var e=0;e<Ls.length;e++)Ls[e]._workInProgressVersionPrimary=null;Ls.length=0}var $o=un.ReactCurrentDispatcher,js=un.ReactCurrentBatchConfig,nr=0,ve=null,Pe=null,je=null,Sa=!1,Si=!1,Xi=0,Ty=0;function Oe(){throw Error(L(321))}function rd(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!Wt(e[n],t[n]))return!1;return!0}function id(e,t,n,r,i,o){if(nr=o,ve=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,$o.current=e===null||e.memoizedState===null?Ey:Py,e=n(r,i),Si){o=0;do{if(Si=!1,Xi=0,25<=o)throw Error(L(301));o+=1,je=Pe=null,t.updateQueue=null,$o.current=Ny,e=n(r,i)}while(Si)}if($o.current=za,t=Pe!==null&&Pe.next!==null,nr=0,je=Pe=ve=null,Sa=!1,t)throw Error(L(300));return e}function od(){var e=Xi!==0;return Xi=0,e}function Ut(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return je===null?ve.memoizedState=je=e:je=je.next=e,je}function zt(){if(Pe===null){var e=ve.alternate;e=e!==null?e.memoizedState:null}else e=Pe.next;var t=je===null?ve.memoizedState:je.next;if(t!==null)je=t,Pe=e;else{if(e===null)throw Error(L(310));Pe=e,e={memoizedState:Pe.memoizedState,baseState:Pe.baseState,baseQueue:Pe.baseQueue,queue:Pe.queue,next:null},je===null?ve.memoizedState=je=e:je=je.next=e}return je}function Ki(e,t){return typeof t=="function"?t(e):t}function Rs(e){var t=zt(),n=t.queue;if(n===null)throw Error(L(311));n.lastRenderedReducer=e;var r=Pe,i=r.baseQueue,o=n.pending;if(o!==null){if(i!==null){var a=i.next;i.next=o.next,o.next=a}r.baseQueue=i=o,n.pending=null}if(i!==null){o=i.next,r=r.baseState;var s=a=null,l=null,u=o;do{var c=u.lane;if((nr&c)===c)l!==null&&(l=l.next={lane:0,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),r=u.hasEagerState?u.eagerState:e(r,u.action);else{var d={lane:c,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null};l===null?(s=l=d,a=r):l=l.next=d,ve.lanes|=c,rr|=c}u=u.next}while(u!==null&&u!==o);l===null?a=r:l.next=s,Wt(r,t.memoizedState)||(it=!0),t.memoizedState=r,t.baseState=a,t.baseQueue=l,n.lastRenderedState=r}if(e=n.interleaved,e!==null){i=e;do o=i.lane,ve.lanes|=o,rr|=o,i=i.next;while(i!==e)}else i===null&&(n.lanes=0);return[t.memoizedState,n.dispatch]}function Fs(e){var t=zt(),n=t.queue;if(n===null)throw Error(L(311));n.lastRenderedReducer=e;var r=n.dispatch,i=n.pending,o=t.memoizedState;if(i!==null){n.pending=null;var a=i=i.next;do o=e(o,a.action),a=a.next;while(a!==i);Wt(o,t.memoizedState)||(it=!0),t.memoizedState=o,t.baseQueue===null&&(t.baseState=o),n.lastRenderedState=o}return[o,r]}function Hm(){}function Qm(e,t){var n=ve,r=zt(),i=t(),o=!Wt(r.memoizedState,i);if(o&&(r.memoizedState=i,it=!0),r=r.queue,ad(Ym.bind(null,n,r,e),[e]),r.getSnapshot!==t||o||je!==null&&je.memoizedState.tag&1){if(n.flags|=2048,Yi(9,Km.bind(null,n,r,i,t),void 0,null),Fe===null)throw Error(L(349));nr&30||Xm(n,t,i)}return i}function Xm(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=ve.updateQueue,t===null?(t={lastEffect:null,stores:null},ve.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function Km(e,t,n,r){t.value=n,t.getSnapshot=r,$m(t)&&Gm(e)}function Ym(e,t,n){return n(function(){$m(t)&&Gm(e)})}function $m(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!Wt(e,n)}catch{return!0}}function Gm(e){var t=sn(e,1);t!==null&&qt(t,e,1,-1)}function Ac(e){var t=Ut();return typeof e=="function"&&(e=e()),t.memoizedState=t.baseState=e,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:Ki,lastRenderedState:e},t.queue=e,e=e.dispatch=zy.bind(null,ve,e),[t.memoizedState,e]}function Yi(e,t,n,r){return e={tag:e,create:t,destroy:n,deps:r,next:null},t=ve.updateQueue,t===null?(t={lastEffect:null,stores:null},ve.updateQueue=t,t.lastEffect=e.next=e):(n=t.lastEffect,n===null?t.lastEffect=e.next=e:(r=n.next,n.next=e,e.next=r,t.lastEffect=e)),e}function Zm(){return zt().memoizedState}function Go(e,t,n,r){var i=Ut();ve.flags|=e,i.memoizedState=Yi(1|t,n,void 0,r===void 0?null:r)}function Xa(e,t,n,r){var i=zt();r=r===void 0?null:r;var o=void 0;if(Pe!==null){var a=Pe.memoizedState;if(o=a.destroy,r!==null&&rd(r,a.deps)){i.memoizedState=Yi(t,n,o,r);return}}ve.flags|=e,i.memoizedState=Yi(1|t,n,o,r)}function Lc(e,t){return Go(8390656,8,e,t)}function ad(e,t){return Xa(2048,8,e,t)}function Jm(e,t){return Xa(4,2,e,t)}function ef(e,t){return Xa(4,4,e,t)}function tf(e,t){if(typeof t=="function")return e=e(),t(e),function(){t(null)};if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function nf(e,t,n){return n=n!=null?n.concat([e]):null,Xa(4,4,tf.bind(null,t,e),n)}function sd(){}function rf(e,t){var n=zt();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&rd(t,r[1])?r[0]:(n.memoizedState=[e,t],e)}function of(e,t){var n=zt();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&rd(t,r[1])?r[0]:(e=e(),n.memoizedState=[e,t],e)}function af(e,t,n){return nr&21?(Wt(n,t)||(n=dm(),ve.lanes|=n,rr|=n,e.baseState=!0),t):(e.baseState&&(e.baseState=!1,it=!0),e.memoizedState=n)}function Cy(e,t){var n=de;de=n!==0&&4>n?n:4,e(!0);var r=js.transition;js.transition={};try{e(!1),t()}finally{de=n,js.transition=r}}function sf(){return zt().memoizedState}function Sy(e,t,n){var r=Ln(e);if(n={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null},lf(e))uf(t,n);else if(n=Vm(e,t,n,r),n!==null){var i=Ge();qt(n,e,r,i),df(n,t,r)}}function zy(e,t,n){var r=Ln(e),i={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null};if(lf(e))uf(t,i);else{var o=e.alternate;if(e.lanes===0&&(o===null||o.lanes===0)&&(o=t.lastRenderedReducer,o!==null))try{var a=t.lastRenderedState,s=o(a,n);if(i.hasEagerState=!0,i.eagerState=s,Wt(s,a)){var l=t.interleaved;l===null?(i.next=i,Zu(t)):(i.next=l.next,l.next=i),t.interleaved=i;return}}catch{}finally{}n=Vm(e,t,i,r),n!==null&&(i=Ge(),qt(n,e,r,i),df(n,t,r))}}function lf(e){var t=e.alternate;return e===ve||t!==null&&t===ve}function uf(e,t){Si=Sa=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function df(e,t,n){if(n&4194240){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,Mu(e,n)}}var za={readContext:St,useCallback:Oe,useContext:Oe,useEffect:Oe,useImperativeHandle:Oe,useInsertionEffect:Oe,useLayoutEffect:Oe,useMemo:Oe,useReducer:Oe,useRef:Oe,useState:Oe,useDebugValue:Oe,useDeferredValue:Oe,useTransition:Oe,useMutableSource:Oe,useSyncExternalStore:Oe,useId:Oe,unstable_isNewReconciler:!1},Ey={readContext:St,useCallback:function(e,t){return Ut().memoizedState=[e,t===void 0?null:t],e},useContext:St,useEffect:Lc,useImperativeHandle:function(e,t,n){return n=n!=null?n.concat([e]):null,Go(4194308,4,tf.bind(null,t,e),n)},useLayoutEffect:function(e,t){return Go(4194308,4,e,t)},useInsertionEffect:function(e,t){return Go(4,2,e,t)},useMemo:function(e,t){var n=Ut();return t=t===void 0?null:t,e=e(),n.memoizedState=[e,t],e},useReducer:function(e,t,n){var r=Ut();return t=n!==void 0?n(t):t,r.memoizedState=r.baseState=t,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:t},r.queue=e,e=e.dispatch=Sy.bind(null,ve,e),[r.memoizedState,e]},useRef:function(e){var t=Ut();return e={current:e},t.memoizedState=e},useState:Ac,useDebugValue:sd,useDeferredValue:function(e){return Ut().memoizedState=e},useTransition:function(){var e=Ac(!1),t=e[0];return e=Cy.bind(null,e[1]),Ut().memoizedState=e,[t,e]},useMutableSource:function(){},useSyncExternalStore:function(e,t,n){var r=ve,i=Ut();if(ye){if(n===void 0)throw Error(L(407));n=n()}else{if(n=t(),Fe===null)throw Error(L(349));nr&30||Xm(r,t,n)}i.memoizedState=n;var o={value:n,getSnapshot:t};return i.queue=o,Lc(Ym.bind(null,r,o,e),[e]),r.flags|=2048,Yi(9,Km.bind(null,r,o,n,t),void 0,null),n},useId:function(){var e=Ut(),t=Fe.identifierPrefix;if(ye){var n=nn,r=tn;n=(r&~(1<<32-Mt(r)-1)).toString(32)+n,t=":"+t+"R"+n,n=Xi++,0<n&&(t+="H"+n.toString(32)),t+=":"}else n=Ty++,t=":"+t+"r"+n.toString(32)+":";return e.memoizedState=t},unstable_isNewReconciler:!1},Py={readContext:St,useCallback:rf,useContext:St,useEffect:ad,useImperativeHandle:nf,useInsertionEffect:Jm,useLayoutEffect:ef,useMemo:of,useReducer:Rs,useRef:Zm,useState:function(){return Rs(Ki)},useDebugValue:sd,useDeferredValue:function(e){var t=zt();return af(t,Pe.memoizedState,e)},useTransition:function(){var e=Rs(Ki)[0],t=zt().memoizedState;return[e,t]},useMutableSource:Hm,useSyncExternalStore:Qm,useId:sf,unstable_isNewReconciler:!1},Ny={readContext:St,useCallback:rf,useContext:St,useEffect:ad,useImperativeHandle:nf,useInsertionEffect:Jm,useLayoutEffect:ef,useMemo:of,useReducer:Fs,useRef:Zm,useState:function(){return Fs(Ki)},useDebugValue:sd,useDeferredValue:function(e){var t=zt();return Pe===null?t.memoizedState=e:af(t,Pe.memoizedState,e)},useTransition:function(){var e=Fs(Ki)[0],t=zt().memoizedState;return[e,t]},useMutableSource:Hm,useSyncExternalStore:Qm,useId:sf,unstable_isNewReconciler:!1};function Rt(e,t){if(e&&e.defaultProps){t=we({},t),e=e.defaultProps;for(var n in e)t[n]===void 0&&(t[n]=e[n]);return t}return t}function Bl(e,t,n,r){t=e.memoizedState,n=n(r,t),n=n==null?t:we({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var Ka={isMounted:function(e){return(e=e._reactInternals)?lr(e)===e:!1},enqueueSetState:function(e,t,n){e=e._reactInternals;var r=Ge(),i=Ln(e),o=rn(r,i);o.payload=t,n!=null&&(o.callback=n),t=Nn(e,o,i),t!==null&&(qt(t,e,i,r),Yo(t,e,i))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var r=Ge(),i=Ln(e),o=rn(r,i);o.tag=1,o.payload=t,n!=null&&(o.callback=n),t=Nn(e,o,i),t!==null&&(qt(t,e,i,r),Yo(t,e,i))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=Ge(),r=Ln(e),i=rn(n,r);i.tag=2,t!=null&&(i.callback=t),t=Nn(e,i,r),t!==null&&(qt(t,e,r,n),Yo(t,e,r))}};function jc(e,t,n,r,i,o,a){return e=e.stateNode,typeof e.shouldComponentUpdate=="function"?e.shouldComponentUpdate(r,o,a):t.prototype&&t.prototype.isPureReactComponent?!Wi(n,r)||!Wi(i,o):!0}function cf(e,t,n){var r=!1,i=Dn,o=t.contextType;return typeof o=="object"&&o!==null?o=St(o):(i=at(t)?er:He.current,r=t.contextTypes,o=(r=r!=null)?Wr(e,i):Dn),t=new t(n,o),e.memoizedState=t.state!==null&&t.state!==void 0?t.state:null,t.updater=Ka,e.stateNode=t,t._reactInternals=e,r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=i,e.__reactInternalMemoizedMaskedChildContext=o),t}function Rc(e,t,n,r){e=t.state,typeof t.componentWillReceiveProps=="function"&&t.componentWillReceiveProps(n,r),typeof t.UNSAFE_componentWillReceiveProps=="function"&&t.UNSAFE_componentWillReceiveProps(n,r),t.state!==e&&Ka.enqueueReplaceState(t,t.state,null)}function Ul(e,t,n,r){var i=e.stateNode;i.props=n,i.state=e.memoizedState,i.refs={},Ju(e);var o=t.contextType;typeof o=="object"&&o!==null?i.context=St(o):(o=at(t)?er:He.current,i.context=Wr(e,o)),i.state=e.memoizedState,o=t.getDerivedStateFromProps,typeof o=="function"&&(Bl(e,t,o,n),i.state=e.memoizedState),typeof t.getDerivedStateFromProps=="function"||typeof i.getSnapshotBeforeUpdate=="function"||typeof i.UNSAFE_componentWillMount!="function"&&typeof i.componentWillMount!="function"||(t=i.state,typeof i.componentWillMount=="function"&&i.componentWillMount(),typeof i.UNSAFE_componentWillMount=="function"&&i.UNSAFE_componentWillMount(),t!==i.state&&Ka.enqueueReplaceState(i,i.state,null),Ta(e,n,i,r),i.state=e.memoizedState),typeof i.componentDidMount=="function"&&(e.flags|=4194308)}function Hr(e,t){try{var n="",r=t;do n+=i_(r),r=r.return;while(r);var i=n}catch(o){i=`
Error generating stack: `+o.message+`
`+o.stack}return{value:e,source:t,stack:i,digest:null}}function Is(e,t,n){return{value:e,source:null,stack:n??null,digest:t??null}}function Hl(e,t){try{console.error(t.value)}catch(n){setTimeout(function(){throw n})}}var Ay=typeof WeakMap=="function"?WeakMap:Map;function pf(e,t,n){n=rn(-1,n),n.tag=3,n.payload={element:null};var r=t.value;return n.callback=function(){Pa||(Pa=!0,tu=r),Hl(e,t)},n}function hf(e,t,n){n=rn(-1,n),n.tag=3;var r=e.type.getDerivedStateFromError;if(typeof r=="function"){var i=t.value;n.payload=function(){return r(i)},n.callback=function(){Hl(e,t)}}var o=e.stateNode;return o!==null&&typeof o.componentDidCatch=="function"&&(n.callback=function(){Hl(e,t),typeof r!="function"&&(An===null?An=new Set([this]):An.add(this));var a=t.stack;this.componentDidCatch(t.value,{componentStack:a!==null?a:""})}),n}function Fc(e,t,n){var r=e.pingCache;if(r===null){r=e.pingCache=new Ay;var i=new Set;r.set(t,i)}else i=r.get(t),i===void 0&&(i=new Set,r.set(t,i));i.has(n)||(i.add(n),e=Hy.bind(null,e,t,n),t.then(e,e))}function Ic(e){do{var t;if((t=e.tag===13)&&(t=e.memoizedState,t=t!==null?t.dehydrated!==null:!0),t)return e;e=e.return}while(e!==null);return null}function Dc(e,t,n,r,i){return e.mode&1?(e.flags|=65536,e.lanes=i,e):(e===t?e.flags|=65536:(e.flags|=128,n.flags|=131072,n.flags&=-52805,n.tag===1&&(n.alternate===null?n.tag=17:(t=rn(-1,1),t.tag=2,Nn(n,t,1))),n.lanes|=1),e)}var Ly=un.ReactCurrentOwner,it=!1;function Ke(e,t,n,r){t.child=e===null?Wm(t,null,n,r):Br(t,e.child,n,r)}function Mc(e,t,n,r,i){n=n.render;var o=t.ref;return Rr(t,i),r=id(e,t,n,r,o,i),n=od(),e!==null&&!it?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~i,ln(e,t,i)):(ye&&n&&Qu(t),t.flags|=1,Ke(e,t,r,i),t.child)}function qc(e,t,n,r,i){if(e===null){var o=n.type;return typeof o=="function"&&!fd(o)&&o.defaultProps===void 0&&n.compare===null&&n.defaultProps===void 0?(t.tag=15,t.type=o,mf(e,t,o,r,i)):(e=ta(n.type,null,r,t,t.mode,i),e.ref=t.ref,e.return=t,t.child=e)}if(o=e.child,!(e.lanes&i)){var a=o.memoizedProps;if(n=n.compare,n=n!==null?n:Wi,n(a,r)&&e.ref===t.ref)return ln(e,t,i)}return t.flags|=1,e=jn(o,r),e.ref=t.ref,e.return=t,t.child=e}function mf(e,t,n,r,i){if(e!==null){var o=e.memoizedProps;if(Wi(o,r)&&e.ref===t.ref)if(it=!1,t.pendingProps=r=o,(e.lanes&i)!==0)e.flags&131072&&(it=!0);else return t.lanes=e.lanes,ln(e,t,i)}return Ql(e,t,n,r,i)}function ff(e,t,n){var r=t.pendingProps,i=r.children,o=e!==null?e.memoizedState:null;if(r.mode==="hidden")if(!(t.mode&1))t.memoizedState={baseLanes:0,cachePool:null,transitions:null},me(Er,dt),dt|=n;else{if(!(n&1073741824))return e=o!==null?o.baseLanes|n:n,t.lanes=t.childLanes=1073741824,t.memoizedState={baseLanes:e,cachePool:null,transitions:null},t.updateQueue=null,me(Er,dt),dt|=e,null;t.memoizedState={baseLanes:0,cachePool:null,transitions:null},r=o!==null?o.baseLanes:n,me(Er,dt),dt|=r}else o!==null?(r=o.baseLanes|n,t.memoizedState=null):r=n,me(Er,dt),dt|=r;return Ke(e,t,i,n),t.child}function gf(e,t){var n=t.ref;(e===null&&n!==null||e!==null&&e.ref!==n)&&(t.flags|=512,t.flags|=2097152)}function Ql(e,t,n,r,i){var o=at(n)?er:He.current;return o=Wr(t,o),Rr(t,i),n=id(e,t,n,r,o,i),r=od(),e!==null&&!it?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~i,ln(e,t,i)):(ye&&r&&Qu(t),t.flags|=1,Ke(e,t,n,i),t.child)}function Oc(e,t,n,r,i){if(at(n)){var o=!0;xa(t)}else o=!1;if(Rr(t,i),t.stateNode===null)Zo(e,t),cf(t,n,r),Ul(t,n,r,i),r=!0;else if(e===null){var a=t.stateNode,s=t.memoizedProps;a.props=s;var l=a.context,u=n.contextType;typeof u=="object"&&u!==null?u=St(u):(u=at(n)?er:He.current,u=Wr(t,u));var c=n.getDerivedStateFromProps,d=typeof c=="function"||typeof a.getSnapshotBeforeUpdate=="function";d||typeof a.UNSAFE_componentWillReceiveProps!="function"&&typeof a.componentWillReceiveProps!="function"||(s!==r||l!==u)&&Rc(t,a,r,u),xn=!1;var h=t.memoizedState;a.state=h,Ta(t,r,a,i),l=t.memoizedState,s!==r||h!==l||ot.current||xn?(typeof c=="function"&&(Bl(t,n,c,r),l=t.memoizedState),(s=xn||jc(t,n,s,r,h,l,u))?(d||typeof a.UNSAFE_componentWillMount!="function"&&typeof a.componentWillMount!="function"||(typeof a.componentWillMount=="function"&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount=="function"&&a.UNSAFE_componentWillMount()),typeof a.componentDidMount=="function"&&(t.flags|=4194308)):(typeof a.componentDidMount=="function"&&(t.flags|=4194308),t.memoizedProps=r,t.memoizedState=l),a.props=r,a.state=l,a.context=u,r=s):(typeof a.componentDidMount=="function"&&(t.flags|=4194308),r=!1)}else{a=t.stateNode,Bm(e,t),s=t.memoizedProps,u=t.type===t.elementType?s:Rt(t.type,s),a.props=u,d=t.pendingProps,h=a.context,l=n.contextType,typeof l=="object"&&l!==null?l=St(l):(l=at(n)?er:He.current,l=Wr(t,l));var m=n.getDerivedStateFromProps;(c=typeof m=="function"||typeof a.getSnapshotBeforeUpdate=="function")||typeof a.UNSAFE_componentWillReceiveProps!="function"&&typeof a.componentWillReceiveProps!="function"||(s!==d||h!==l)&&Rc(t,a,r,l),xn=!1,h=t.memoizedState,a.state=h,Ta(t,r,a,i);var f=t.memoizedState;s!==d||h!==f||ot.current||xn?(typeof m=="function"&&(Bl(t,n,m,r),f=t.memoizedState),(u=xn||jc(t,n,u,r,h,f,l)||!1)?(c||typeof a.UNSAFE_componentWillUpdate!="function"&&typeof a.componentWillUpdate!="function"||(typeof a.componentWillUpdate=="function"&&a.componentWillUpdate(r,f,l),typeof a.UNSAFE_componentWillUpdate=="function"&&a.UNSAFE_componentWillUpdate(r,f,l)),typeof a.componentDidUpdate=="function"&&(t.flags|=4),typeof a.getSnapshotBeforeUpdate=="function"&&(t.flags|=1024)):(typeof a.componentDidUpdate!="function"||s===e.memoizedProps&&h===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!="function"||s===e.memoizedProps&&h===e.memoizedState||(t.flags|=1024),t.memoizedProps=r,t.memoizedState=f),a.props=r,a.state=f,a.context=l,r=u):(typeof a.componentDidUpdate!="function"||s===e.memoizedProps&&h===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!="function"||s===e.memoizedProps&&h===e.memoizedState||(t.flags|=1024),r=!1)}return Xl(e,t,n,r,o,i)}function Xl(e,t,n,r,i,o){gf(e,t);var a=(t.flags&128)!==0;if(!r&&!a)return i&&Cc(t,n,!1),ln(e,t,o);r=t.stateNode,Ly.current=t;var s=a&&typeof n.getDerivedStateFromError!="function"?null:r.render();return t.flags|=1,e!==null&&a?(t.child=Br(t,e.child,null,o),t.child=Br(t,null,s,o)):Ke(e,t,s,o),t.memoizedState=r.state,i&&Cc(t,n,!0),t.child}function _f(e){var t=e.stateNode;t.pendingContext?Tc(e,t.pendingContext,t.pendingContext!==t.context):t.context&&Tc(e,t.context,!1),ed(e,t.containerInfo)}function Wc(e,t,n,r,i){return Vr(),Ku(i),t.flags|=256,Ke(e,t,n,r),t.child}var Kl={dehydrated:null,treeContext:null,retryLane:0};function Yl(e){return{baseLanes:e,cachePool:null,transitions:null}}function yf(e,t,n){var r=t.pendingProps,i=be.current,o=!1,a=(t.flags&128)!==0,s;if((s=a)||(s=e!==null&&e.memoizedState===null?!1:(i&2)!==0),s?(o=!0,t.flags&=-129):(e===null||e.memoizedState!==null)&&(i|=1),me(be,i&1),e===null)return Wl(t),e=t.memoizedState,e!==null&&(e=e.dehydrated,e!==null)?(t.mode&1?e.data==="$!"?t.lanes=8:t.lanes=1073741824:t.lanes=1,null):(a=r.children,e=r.fallback,o?(r=t.mode,o=t.child,a={mode:"hidden",children:a},!(r&1)&&o!==null?(o.childLanes=0,o.pendingProps=a):o=Ga(a,r,0,null),e=Jn(e,r,n,null),o.return=t,e.return=t,o.sibling=e,t.child=o,t.child.memoizedState=Yl(n),t.memoizedState=Kl,e):ld(t,a));if(i=e.memoizedState,i!==null&&(s=i.dehydrated,s!==null))return jy(e,t,a,r,s,i,n);if(o){o=r.fallback,a=t.mode,i=e.child,s=i.sibling;var l={mode:"hidden",children:r.children};return!(a&1)&&t.child!==i?(r=t.child,r.childLanes=0,r.pendingProps=l,t.deletions=null):(r=jn(i,l),r.subtreeFlags=i.subtreeFlags&14680064),s!==null?o=jn(s,o):(o=Jn(o,a,n,null),o.flags|=2),o.return=t,r.return=t,r.sibling=o,t.child=r,r=o,o=t.child,a=e.child.memoizedState,a=a===null?Yl(n):{baseLanes:a.baseLanes|n,cachePool:null,transitions:a.transitions},o.memoizedState=a,o.childLanes=e.childLanes&~n,t.memoizedState=Kl,r}return o=e.child,e=o.sibling,r=jn(o,{mode:"visible",children:r.children}),!(t.mode&1)&&(r.lanes=n),r.return=t,r.sibling=null,e!==null&&(n=t.deletions,n===null?(t.deletions=[e],t.flags|=16):n.push(e)),t.child=r,t.memoizedState=null,r}function ld(e,t){return t=Ga({mode:"visible",children:t},e.mode,0,null),t.return=e,e.child=t}function Ao(e,t,n,r){return r!==null&&Ku(r),Br(t,e.child,null,n),e=ld(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function jy(e,t,n,r,i,o,a){if(n)return t.flags&256?(t.flags&=-257,r=Is(Error(L(422))),Ao(e,t,a,r)):t.memoizedState!==null?(t.child=e.child,t.flags|=128,null):(o=r.fallback,i=t.mode,r=Ga({mode:"visible",children:r.children},i,0,null),o=Jn(o,i,a,null),o.flags|=2,r.return=t,o.return=t,r.sibling=o,t.child=r,t.mode&1&&Br(t,e.child,null,a),t.child.memoizedState=Yl(a),t.memoizedState=Kl,o);if(!(t.mode&1))return Ao(e,t,a,null);if(i.data==="$!"){if(r=i.nextSibling&&i.nextSibling.dataset,r)var s=r.dgst;return r=s,o=Error(L(419)),r=Is(o,r,void 0),Ao(e,t,a,r)}if(s=(a&e.childLanes)!==0,it||s){if(r=Fe,r!==null){switch(a&-a){case 4:i=2;break;case 16:i=8;break;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:i=32;break;case 536870912:i=268435456;break;default:i=0}i=i&(r.suspendedLanes|a)?0:i,i!==0&&i!==o.retryLane&&(o.retryLane=i,sn(e,i),qt(r,e,i,-1))}return md(),r=Is(Error(L(421))),Ao(e,t,a,r)}return i.data==="$?"?(t.flags|=128,t.child=e.child,t=Qy.bind(null,e),i._reactRetry=t,null):(e=o.treeContext,pt=Pn(i.nextSibling),ht=t,ye=!0,It=null,e!==null&&(vt[kt++]=tn,vt[kt++]=nn,vt[kt++]=tr,tn=e.id,nn=e.overflow,tr=t),t=ld(t,r.children),t.flags|=4096,t)}function Vc(e,t,n){e.lanes|=t;var r=e.alternate;r!==null&&(r.lanes|=t),Vl(e.return,t,n)}function Ds(e,t,n,r,i){var o=e.memoizedState;o===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:r,tail:n,tailMode:i}:(o.isBackwards=t,o.rendering=null,o.renderingStartTime=0,o.last=r,o.tail=n,o.tailMode=i)}function xf(e,t,n){var r=t.pendingProps,i=r.revealOrder,o=r.tail;if(Ke(e,t,r.children,n),r=be.current,r&2)r=r&1|2,t.flags|=128;else{if(e!==null&&e.flags&128)e:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&Vc(e,n,t);else if(e.tag===19)Vc(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break e;for(;e.sibling===null;){if(e.return===null||e.return===t)break e;e=e.return}e.sibling.return=e.return,e=e.sibling}r&=1}if(me(be,r),!(t.mode&1))t.memoizedState=null;else switch(i){case"forwards":for(n=t.child,i=null;n!==null;)e=n.alternate,e!==null&&Ca(e)===null&&(i=n),n=n.sibling;n=i,n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null),Ds(t,!1,i,n,o);break;case"backwards":for(n=null,i=t.child,t.child=null;i!==null;){if(e=i.alternate,e!==null&&Ca(e)===null){t.child=i;break}e=i.sibling,i.sibling=n,n=i,i=e}Ds(t,!0,n,null,o);break;case"together":Ds(t,!1,null,null,void 0);break;default:t.memoizedState=null}return t.child}function Zo(e,t){!(t.mode&1)&&e!==null&&(e.alternate=null,t.alternate=null,t.flags|=2)}function ln(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),rr|=t.lanes,!(n&t.childLanes))return null;if(e!==null&&t.child!==e.child)throw Error(L(153));if(t.child!==null){for(e=t.child,n=jn(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=jn(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function Ry(e,t,n){switch(t.tag){case 3:_f(t),Vr();break;case 5:Um(t);break;case 1:at(t.type)&&xa(t);break;case 4:ed(t,t.stateNode.containerInfo);break;case 10:var r=t.type._context,i=t.memoizedProps.value;me(ka,r._currentValue),r._currentValue=i;break;case 13:if(r=t.memoizedState,r!==null)return r.dehydrated!==null?(me(be,be.current&1),t.flags|=128,null):n&t.child.childLanes?yf(e,t,n):(me(be,be.current&1),e=ln(e,t,n),e!==null?e.sibling:null);me(be,be.current&1);break;case 19:if(r=(n&t.childLanes)!==0,e.flags&128){if(r)return xf(e,t,n);t.flags|=128}if(i=t.memoizedState,i!==null&&(i.rendering=null,i.tail=null,i.lastEffect=null),me(be,be.current),r)break;return null;case 22:case 23:return t.lanes=0,ff(e,t,n)}return ln(e,t,n)}var bf,$l,vf,kf;bf=function(e,t){for(var n=t.child;n!==null;){if(n.tag===5||n.tag===6)e.appendChild(n.stateNode);else if(n.tag!==4&&n.child!==null){n.child.return=n,n=n.child;continue}if(n===t)break;for(;n.sibling===null;){if(n.return===null||n.return===t)return;n=n.return}n.sibling.return=n.return,n=n.sibling}};$l=function(){};vf=function(e,t,n,r){var i=e.memoizedProps;if(i!==r){e=t.stateNode,$n(Kt.current);var o=null;switch(n){case"input":i=yl(e,i),r=yl(e,r),o=[];break;case"select":i=we({},i,{value:void 0}),r=we({},r,{value:void 0}),o=[];break;case"textarea":i=vl(e,i),r=vl(e,r),o=[];break;default:typeof i.onClick!="function"&&typeof r.onClick=="function"&&(e.onclick=_a)}wl(n,r);var a;n=null;for(u in i)if(!r.hasOwnProperty(u)&&i.hasOwnProperty(u)&&i[u]!=null)if(u==="style"){var s=i[u];for(a in s)s.hasOwnProperty(a)&&(n||(n={}),n[a]="")}else u!=="dangerouslySetInnerHTML"&&u!=="children"&&u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&u!=="autoFocus"&&(Ri.hasOwnProperty(u)?o||(o=[]):(o=o||[]).push(u,null));for(u in r){var l=r[u];if(s=i!=null?i[u]:void 0,r.hasOwnProperty(u)&&l!==s&&(l!=null||s!=null))if(u==="style")if(s){for(a in s)!s.hasOwnProperty(a)||l&&l.hasOwnProperty(a)||(n||(n={}),n[a]="");for(a in l)l.hasOwnProperty(a)&&s[a]!==l[a]&&(n||(n={}),n[a]=l[a])}else n||(o||(o=[]),o.push(u,n)),n=l;else u==="dangerouslySetInnerHTML"?(l=l?l.__html:void 0,s=s?s.__html:void 0,l!=null&&s!==l&&(o=o||[]).push(u,l)):u==="children"?typeof l!="string"&&typeof l!="number"||(o=o||[]).push(u,""+l):u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&(Ri.hasOwnProperty(u)?(l!=null&&u==="onScroll"&&ge("scroll",e),o||s===l||(o=[])):(o=o||[]).push(u,l))}n&&(o=o||[]).push("style",n);var u=o;(t.updateQueue=u)&&(t.flags|=4)}};kf=function(e,t,n,r){n!==r&&(t.flags|=4)};function ui(e,t){if(!ye)switch(e.tailMode){case"hidden":t=e.tail;for(var n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null;break;case"collapsed":n=e.tail;for(var r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:r.sibling=null}}function We(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,r=0;if(t)for(var i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags&14680064,r|=i.flags&14680064,i.return=e,i=i.sibling;else for(i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags,r|=i.flags,i.return=e,i=i.sibling;return e.subtreeFlags|=r,e.childLanes=n,t}function Fy(e,t,n){var r=t.pendingProps;switch(Xu(t),t.tag){case 2:case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return We(t),null;case 1:return at(t.type)&&ya(),We(t),null;case 3:return r=t.stateNode,Ur(),_e(ot),_e(He),nd(),r.pendingContext&&(r.context=r.pendingContext,r.pendingContext=null),(e===null||e.child===null)&&(Po(t)?t.flags|=4:e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,It!==null&&(iu(It),It=null))),$l(e,t),We(t),null;case 5:td(t);var i=$n(Qi.current);if(n=t.type,e!==null&&t.stateNode!=null)vf(e,t,n,r,i),e.ref!==t.ref&&(t.flags|=512,t.flags|=2097152);else{if(!r){if(t.stateNode===null)throw Error(L(166));return We(t),null}if(e=$n(Kt.current),Po(t)){r=t.stateNode,n=t.type;var o=t.memoizedProps;switch(r[Qt]=t,r[Ui]=o,e=(t.mode&1)!==0,n){case"dialog":ge("cancel",r),ge("close",r);break;case"iframe":case"object":case"embed":ge("load",r);break;case"video":case"audio":for(i=0;i<xi.length;i++)ge(xi[i],r);break;case"source":ge("error",r);break;case"img":case"image":case"link":ge("error",r),ge("load",r);break;case"details":ge("toggle",r);break;case"input":Gd(r,o),ge("invalid",r);break;case"select":r._wrapperState={wasMultiple:!!o.multiple},ge("invalid",r);break;case"textarea":Jd(r,o),ge("invalid",r)}wl(n,o),i=null;for(var a in o)if(o.hasOwnProperty(a)){var s=o[a];a==="children"?typeof s=="string"?r.textContent!==s&&(o.suppressHydrationWarning!==!0&&Eo(r.textContent,s,e),i=["children",s]):typeof s=="number"&&r.textContent!==""+s&&(o.suppressHydrationWarning!==!0&&Eo(r.textContent,s,e),i=["children",""+s]):Ri.hasOwnProperty(a)&&s!=null&&a==="onScroll"&&ge("scroll",r)}switch(n){case"input":bo(r),Zd(r,o,!0);break;case"textarea":bo(r),ec(r);break;case"select":case"option":break;default:typeof o.onClick=="function"&&(r.onclick=_a)}r=i,t.updateQueue=r,r!==null&&(t.flags|=4)}else{a=i.nodeType===9?i:i.ownerDocument,e==="http://www.w3.org/1999/xhtml"&&(e=Yh(n)),e==="http://www.w3.org/1999/xhtml"?n==="script"?(e=a.createElement("div"),e.innerHTML="<script><\/script>",e=e.removeChild(e.firstChild)):typeof r.is=="string"?e=a.createElement(n,{is:r.is}):(e=a.createElement(n),n==="select"&&(a=e,r.multiple?a.multiple=!0:r.size&&(a.size=r.size))):e=a.createElementNS(e,n),e[Qt]=t,e[Ui]=r,bf(e,t,!1,!1),t.stateNode=e;e:{switch(a=Tl(n,r),n){case"dialog":ge("cancel",e),ge("close",e),i=r;break;case"iframe":case"object":case"embed":ge("load",e),i=r;break;case"video":case"audio":for(i=0;i<xi.length;i++)ge(xi[i],e);i=r;break;case"source":ge("error",e),i=r;break;case"img":case"image":case"link":ge("error",e),ge("load",e),i=r;break;case"details":ge("toggle",e),i=r;break;case"input":Gd(e,r),i=yl(e,r),ge("invalid",e);break;case"option":i=r;break;case"select":e._wrapperState={wasMultiple:!!r.multiple},i=we({},r,{value:void 0}),ge("invalid",e);break;case"textarea":Jd(e,r),i=vl(e,r),ge("invalid",e);break;default:i=r}wl(n,i),s=i;for(o in s)if(s.hasOwnProperty(o)){var l=s[o];o==="style"?Zh(e,l):o==="dangerouslySetInnerHTML"?(l=l?l.__html:void 0,l!=null&&$h(e,l)):o==="children"?typeof l=="string"?(n!=="textarea"||l!=="")&&Fi(e,l):typeof l=="number"&&Fi(e,""+l):o!=="suppressContentEditableWarning"&&o!=="suppressHydrationWarning"&&o!=="autoFocus"&&(Ri.hasOwnProperty(o)?l!=null&&o==="onScroll"&&ge("scroll",e):l!=null&&Lu(e,o,l,a))}switch(n){case"input":bo(e),Zd(e,r,!1);break;case"textarea":bo(e),ec(e);break;case"option":r.value!=null&&e.setAttribute("value",""+In(r.value));break;case"select":e.multiple=!!r.multiple,o=r.value,o!=null?Nr(e,!!r.multiple,o,!1):r.defaultValue!=null&&Nr(e,!!r.multiple,r.defaultValue,!0);break;default:typeof i.onClick=="function"&&(e.onclick=_a)}switch(n){case"button":case"input":case"select":case"textarea":r=!!r.autoFocus;break e;case"img":r=!0;break e;default:r=!1}}r&&(t.flags|=4)}t.ref!==null&&(t.flags|=512,t.flags|=2097152)}return We(t),null;case 6:if(e&&t.stateNode!=null)kf(e,t,e.memoizedProps,r);else{if(typeof r!="string"&&t.stateNode===null)throw Error(L(166));if(n=$n(Qi.current),$n(Kt.current),Po(t)){if(r=t.stateNode,n=t.memoizedProps,r[Qt]=t,(o=r.nodeValue!==n)&&(e=ht,e!==null))switch(e.tag){case 3:Eo(r.nodeValue,n,(e.mode&1)!==0);break;case 5:e.memoizedProps.suppressHydrationWarning!==!0&&Eo(r.nodeValue,n,(e.mode&1)!==0)}o&&(t.flags|=4)}else r=(n.nodeType===9?n:n.ownerDocument).createTextNode(r),r[Qt]=t,t.stateNode=r}return We(t),null;case 13:if(_e(be),r=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(ye&&pt!==null&&t.mode&1&&!(t.flags&128))qm(),Vr(),t.flags|=98560,o=!1;else if(o=Po(t),r!==null&&r.dehydrated!==null){if(e===null){if(!o)throw Error(L(318));if(o=t.memoizedState,o=o!==null?o.dehydrated:null,!o)throw Error(L(317));o[Qt]=t}else Vr(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;We(t),o=!1}else It!==null&&(iu(It),It=null),o=!0;if(!o)return t.flags&65536?t:null}return t.flags&128?(t.lanes=n,t):(r=r!==null,r!==(e!==null&&e.memoizedState!==null)&&r&&(t.child.flags|=8192,t.mode&1&&(e===null||be.current&1?Ne===0&&(Ne=3):md())),t.updateQueue!==null&&(t.flags|=4),We(t),null);case 4:return Ur(),$l(e,t),e===null&&Vi(t.stateNode.containerInfo),We(t),null;case 10:return Gu(t.type._context),We(t),null;case 17:return at(t.type)&&ya(),We(t),null;case 19:if(_e(be),o=t.memoizedState,o===null)return We(t),null;if(r=(t.flags&128)!==0,a=o.rendering,a===null)if(r)ui(o,!1);else{if(Ne!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(a=Ca(e),a!==null){for(t.flags|=128,ui(o,!1),r=a.updateQueue,r!==null&&(t.updateQueue=r,t.flags|=4),t.subtreeFlags=0,r=n,n=t.child;n!==null;)o=n,e=r,o.flags&=14680066,a=o.alternate,a===null?(o.childLanes=0,o.lanes=e,o.child=null,o.subtreeFlags=0,o.memoizedProps=null,o.memoizedState=null,o.updateQueue=null,o.dependencies=null,o.stateNode=null):(o.childLanes=a.childLanes,o.lanes=a.lanes,o.child=a.child,o.subtreeFlags=0,o.deletions=null,o.memoizedProps=a.memoizedProps,o.memoizedState=a.memoizedState,o.updateQueue=a.updateQueue,o.type=a.type,e=a.dependencies,o.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),n=n.sibling;return me(be,be.current&1|2),t.child}e=e.sibling}o.tail!==null&&Se()>Qr&&(t.flags|=128,r=!0,ui(o,!1),t.lanes=4194304)}else{if(!r)if(e=Ca(a),e!==null){if(t.flags|=128,r=!0,n=e.updateQueue,n!==null&&(t.updateQueue=n,t.flags|=4),ui(o,!0),o.tail===null&&o.tailMode==="hidden"&&!a.alternate&&!ye)return We(t),null}else 2*Se()-o.renderingStartTime>Qr&&n!==1073741824&&(t.flags|=128,r=!0,ui(o,!1),t.lanes=4194304);o.isBackwards?(a.sibling=t.child,t.child=a):(n=o.last,n!==null?n.sibling=a:t.child=a,o.last=a)}return o.tail!==null?(t=o.tail,o.rendering=t,o.tail=t.sibling,o.renderingStartTime=Se(),t.sibling=null,n=be.current,me(be,r?n&1|2:n&1),t):(We(t),null);case 22:case 23:return hd(),r=t.memoizedState!==null,e!==null&&e.memoizedState!==null!==r&&(t.flags|=8192),r&&t.mode&1?dt&1073741824&&(We(t),t.subtreeFlags&6&&(t.flags|=8192)):We(t),null;case 24:return null;case 25:return null}throw Error(L(156,t.tag))}function Iy(e,t){switch(Xu(t),t.tag){case 1:return at(t.type)&&ya(),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return Ur(),_e(ot),_e(He),nd(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 5:return td(t),null;case 13:if(_e(be),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(L(340));Vr()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return _e(be),null;case 4:return Ur(),null;case 10:return Gu(t.type._context),null;case 22:case 23:return hd(),null;case 24:return null;default:return null}}var Lo=!1,Be=!1,Dy=typeof WeakSet=="function"?WeakSet:Set,B=null;function zr(e,t){var n=e.ref;if(n!==null)if(typeof n=="function")try{n(null)}catch(r){Te(e,t,r)}else n.current=null}function Gl(e,t,n){try{n()}catch(r){Te(e,t,r)}}var Bc=!1;function My(e,t){if(Rl=ma,e=Sm(),Hu(e)){if("selectionStart"in e)var n={start:e.selectionStart,end:e.selectionEnd};else e:{n=(n=e.ownerDocument)&&n.defaultView||window;var r=n.getSelection&&n.getSelection();if(r&&r.rangeCount!==0){n=r.anchorNode;var i=r.anchorOffset,o=r.focusNode;r=r.focusOffset;try{n.nodeType,o.nodeType}catch{n=null;break e}var a=0,s=-1,l=-1,u=0,c=0,d=e,h=null;t:for(;;){for(var m;d!==n||i!==0&&d.nodeType!==3||(s=a+i),d!==o||r!==0&&d.nodeType!==3||(l=a+r),d.nodeType===3&&(a+=d.nodeValue.length),(m=d.firstChild)!==null;)h=d,d=m;for(;;){if(d===e)break t;if(h===n&&++u===i&&(s=a),h===o&&++c===r&&(l=a),(m=d.nextSibling)!==null)break;d=h,h=d.parentNode}d=m}n=s===-1||l===-1?null:{start:s,end:l}}else n=null}n=n||{start:0,end:0}}else n=null;for(Fl={focusedElem:e,selectionRange:n},ma=!1,B=t;B!==null;)if(t=B,e=t.child,(t.subtreeFlags&1028)!==0&&e!==null)e.return=t,B=e;else for(;B!==null;){t=B;try{var f=t.alternate;if(t.flags&1024)switch(t.tag){case 0:case 11:case 15:break;case 1:if(f!==null){var x=f.memoizedProps,S=f.memoizedState,g=t.stateNode,y=g.getSnapshotBeforeUpdate(t.elementType===t.type?x:Rt(t.type,x),S);g.__reactInternalSnapshotBeforeUpdate=y}break;case 3:var _=t.stateNode.containerInfo;_.nodeType===1?_.textContent="":_.nodeType===9&&_.documentElement&&_.removeChild(_.documentElement);break;case 5:case 6:case 4:case 17:break;default:throw Error(L(163))}}catch(w){Te(t,t.return,w)}if(e=t.sibling,e!==null){e.return=t.return,B=e;break}B=t.return}return f=Bc,Bc=!1,f}function zi(e,t,n){var r=t.updateQueue;if(r=r!==null?r.lastEffect:null,r!==null){var i=r=r.next;do{if((i.tag&e)===e){var o=i.destroy;i.destroy=void 0,o!==void 0&&Gl(t,n,o)}i=i.next}while(i!==r)}}function Ya(e,t){if(t=t.updateQueue,t=t!==null?t.lastEffect:null,t!==null){var n=t=t.next;do{if((n.tag&e)===e){var r=n.create;n.destroy=r()}n=n.next}while(n!==t)}}function Zl(e){var t=e.ref;if(t!==null){var n=e.stateNode;switch(e.tag){case 5:e=n;break;default:e=n}typeof t=="function"?t(e):t.current=e}}function wf(e){var t=e.alternate;t!==null&&(e.alternate=null,wf(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&(delete t[Qt],delete t[Ui],delete t[Ml],delete t[by],delete t[vy])),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}function Tf(e){return e.tag===5||e.tag===3||e.tag===4}function Uc(e){e:for(;;){for(;e.sibling===null;){if(e.return===null||Tf(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.flags&2||e.child===null||e.tag===4)continue e;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function Jl(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.nodeType===8?n.parentNode.insertBefore(e,t):n.insertBefore(e,t):(n.nodeType===8?(t=n.parentNode,t.insertBefore(e,n)):(t=n,t.appendChild(e)),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=_a));else if(r!==4&&(e=e.child,e!==null))for(Jl(e,t,n),e=e.sibling;e!==null;)Jl(e,t,n),e=e.sibling}function eu(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.insertBefore(e,t):n.appendChild(e);else if(r!==4&&(e=e.child,e!==null))for(eu(e,t,n),e=e.sibling;e!==null;)eu(e,t,n),e=e.sibling}var De=null,Ft=!1;function fn(e,t,n){for(n=n.child;n!==null;)Cf(e,t,n),n=n.sibling}function Cf(e,t,n){if(Xt&&typeof Xt.onCommitFiberUnmount=="function")try{Xt.onCommitFiberUnmount(Wa,n)}catch{}switch(n.tag){case 5:Be||zr(n,t);case 6:var r=De,i=Ft;De=null,fn(e,t,n),De=r,Ft=i,De!==null&&(Ft?(e=De,n=n.stateNode,e.nodeType===8?e.parentNode.removeChild(n):e.removeChild(n)):De.removeChild(n.stateNode));break;case 18:De!==null&&(Ft?(e=De,n=n.stateNode,e.nodeType===8?Ns(e.parentNode,n):e.nodeType===1&&Ns(e,n),qi(e)):Ns(De,n.stateNode));break;case 4:r=De,i=Ft,De=n.stateNode.containerInfo,Ft=!0,fn(e,t,n),De=r,Ft=i;break;case 0:case 11:case 14:case 15:if(!Be&&(r=n.updateQueue,r!==null&&(r=r.lastEffect,r!==null))){i=r=r.next;do{var o=i,a=o.destroy;o=o.tag,a!==void 0&&(o&2||o&4)&&Gl(n,t,a),i=i.next}while(i!==r)}fn(e,t,n);break;case 1:if(!Be&&(zr(n,t),r=n.stateNode,typeof r.componentWillUnmount=="function"))try{r.props=n.memoizedProps,r.state=n.memoizedState,r.componentWillUnmount()}catch(s){Te(n,t,s)}fn(e,t,n);break;case 21:fn(e,t,n);break;case 22:n.mode&1?(Be=(r=Be)||n.memoizedState!==null,fn(e,t,n),Be=r):fn(e,t,n);break;default:fn(e,t,n)}}function Hc(e){var t=e.updateQueue;if(t!==null){e.updateQueue=null;var n=e.stateNode;n===null&&(n=e.stateNode=new Dy),t.forEach(function(r){var i=Xy.bind(null,e,r);n.has(r)||(n.add(r),r.then(i,i))})}}function Pt(e,t){var n=t.deletions;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];try{var o=e,a=t,s=a;e:for(;s!==null;){switch(s.tag){case 5:De=s.stateNode,Ft=!1;break e;case 3:De=s.stateNode.containerInfo,Ft=!0;break e;case 4:De=s.stateNode.containerInfo,Ft=!0;break e}s=s.return}if(De===null)throw Error(L(160));Cf(o,a,i),De=null,Ft=!1;var l=i.alternate;l!==null&&(l.return=null),i.return=null}catch(u){Te(i,t,u)}}if(t.subtreeFlags&12854)for(t=t.child;t!==null;)Sf(t,e),t=t.sibling}function Sf(e,t){var n=e.alternate,r=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(Pt(t,e),Bt(e),r&4){try{zi(3,e,e.return),Ya(3,e)}catch(x){Te(e,e.return,x)}try{zi(5,e,e.return)}catch(x){Te(e,e.return,x)}}break;case 1:Pt(t,e),Bt(e),r&512&&n!==null&&zr(n,n.return);break;case 5:if(Pt(t,e),Bt(e),r&512&&n!==null&&zr(n,n.return),e.flags&32){var i=e.stateNode;try{Fi(i,"")}catch(x){Te(e,e.return,x)}}if(r&4&&(i=e.stateNode,i!=null)){var o=e.memoizedProps,a=n!==null?n.memoizedProps:o,s=e.type,l=e.updateQueue;if(e.updateQueue=null,l!==null)try{s==="input"&&o.type==="radio"&&o.name!=null&&Xh(i,o),Tl(s,a);var u=Tl(s,o);for(a=0;a<l.length;a+=2){var c=l[a],d=l[a+1];c==="style"?Zh(i,d):c==="dangerouslySetInnerHTML"?$h(i,d):c==="children"?Fi(i,d):Lu(i,c,d,u)}switch(s){case"input":xl(i,o);break;case"textarea":Kh(i,o);break;case"select":var h=i._wrapperState.wasMultiple;i._wrapperState.wasMultiple=!!o.multiple;var m=o.value;m!=null?Nr(i,!!o.multiple,m,!1):h!==!!o.multiple&&(o.defaultValue!=null?Nr(i,!!o.multiple,o.defaultValue,!0):Nr(i,!!o.multiple,o.multiple?[]:"",!1))}i[Ui]=o}catch(x){Te(e,e.return,x)}}break;case 6:if(Pt(t,e),Bt(e),r&4){if(e.stateNode===null)throw Error(L(162));i=e.stateNode,o=e.memoizedProps;try{i.nodeValue=o}catch(x){Te(e,e.return,x)}}break;case 3:if(Pt(t,e),Bt(e),r&4&&n!==null&&n.memoizedState.isDehydrated)try{qi(t.containerInfo)}catch(x){Te(e,e.return,x)}break;case 4:Pt(t,e),Bt(e);break;case 13:Pt(t,e),Bt(e),i=e.child,i.flags&8192&&(o=i.memoizedState!==null,i.stateNode.isHidden=o,!o||i.alternate!==null&&i.alternate.memoizedState!==null||(cd=Se())),r&4&&Hc(e);break;case 22:if(c=n!==null&&n.memoizedState!==null,e.mode&1?(Be=(u=Be)||c,Pt(t,e),Be=u):Pt(t,e),Bt(e),r&8192){if(u=e.memoizedState!==null,(e.stateNode.isHidden=u)&&!c&&e.mode&1)for(B=e,c=e.child;c!==null;){for(d=B=c;B!==null;){switch(h=B,m=h.child,h.tag){case 0:case 11:case 14:case 15:zi(4,h,h.return);break;case 1:zr(h,h.return);var f=h.stateNode;if(typeof f.componentWillUnmount=="function"){r=h,n=h.return;try{t=r,f.props=t.memoizedProps,f.state=t.memoizedState,f.componentWillUnmount()}catch(x){Te(r,n,x)}}break;case 5:zr(h,h.return);break;case 22:if(h.memoizedState!==null){Xc(d);continue}}m!==null?(m.return=h,B=m):Xc(d)}c=c.sibling}e:for(c=null,d=e;;){if(d.tag===5){if(c===null){c=d;try{i=d.stateNode,u?(o=i.style,typeof o.setProperty=="function"?o.setProperty("display","none","important"):o.display="none"):(s=d.stateNode,l=d.memoizedProps.style,a=l!=null&&l.hasOwnProperty("display")?l.display:null,s.style.display=Gh("display",a))}catch(x){Te(e,e.return,x)}}}else if(d.tag===6){if(c===null)try{d.stateNode.nodeValue=u?"":d.memoizedProps}catch(x){Te(e,e.return,x)}}else if((d.tag!==22&&d.tag!==23||d.memoizedState===null||d===e)&&d.child!==null){d.child.return=d,d=d.child;continue}if(d===e)break e;for(;d.sibling===null;){if(d.return===null||d.return===e)break e;c===d&&(c=null),d=d.return}c===d&&(c=null),d.sibling.return=d.return,d=d.sibling}}break;case 19:Pt(t,e),Bt(e),r&4&&Hc(e);break;case 21:break;default:Pt(t,e),Bt(e)}}function Bt(e){var t=e.flags;if(t&2){try{e:{for(var n=e.return;n!==null;){if(Tf(n)){var r=n;break e}n=n.return}throw Error(L(160))}switch(r.tag){case 5:var i=r.stateNode;r.flags&32&&(Fi(i,""),r.flags&=-33);var o=Uc(e);eu(e,o,i);break;case 3:case 4:var a=r.stateNode.containerInfo,s=Uc(e);Jl(e,s,a);break;default:throw Error(L(161))}}catch(l){Te(e,e.return,l)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function qy(e,t,n){B=e,zf(e)}function zf(e,t,n){for(var r=(e.mode&1)!==0;B!==null;){var i=B,o=i.child;if(i.tag===22&&r){var a=i.memoizedState!==null||Lo;if(!a){var s=i.alternate,l=s!==null&&s.memoizedState!==null||Be;s=Lo;var u=Be;if(Lo=a,(Be=l)&&!u)for(B=i;B!==null;)a=B,l=a.child,a.tag===22&&a.memoizedState!==null?Kc(i):l!==null?(l.return=a,B=l):Kc(i);for(;o!==null;)B=o,zf(o),o=o.sibling;B=i,Lo=s,Be=u}Qc(e)}else i.subtreeFlags&8772&&o!==null?(o.return=i,B=o):Qc(e)}}function Qc(e){for(;B!==null;){var t=B;if(t.flags&8772){var n=t.alternate;try{if(t.flags&8772)switch(t.tag){case 0:case 11:case 15:Be||Ya(5,t);break;case 1:var r=t.stateNode;if(t.flags&4&&!Be)if(n===null)r.componentDidMount();else{var i=t.elementType===t.type?n.memoizedProps:Rt(t.type,n.memoizedProps);r.componentDidUpdate(i,n.memoizedState,r.__reactInternalSnapshotBeforeUpdate)}var o=t.updateQueue;o!==null&&Nc(t,o,r);break;case 3:var a=t.updateQueue;if(a!==null){if(n=null,t.child!==null)switch(t.child.tag){case 5:n=t.child.stateNode;break;case 1:n=t.child.stateNode}Nc(t,a,n)}break;case 5:var s=t.stateNode;if(n===null&&t.flags&4){n=s;var l=t.memoizedProps;switch(t.type){case"button":case"input":case"select":case"textarea":l.autoFocus&&n.focus();break;case"img":l.src&&(n.src=l.src)}}break;case 6:break;case 4:break;case 12:break;case 13:if(t.memoizedState===null){var u=t.alternate;if(u!==null){var c=u.memoizedState;if(c!==null){var d=c.dehydrated;d!==null&&qi(d)}}}break;case 19:case 17:case 21:case 22:case 23:case 25:break;default:throw Error(L(163))}Be||t.flags&512&&Zl(t)}catch(h){Te(t,t.return,h)}}if(t===e){B=null;break}if(n=t.sibling,n!==null){n.return=t.return,B=n;break}B=t.return}}function Xc(e){for(;B!==null;){var t=B;if(t===e){B=null;break}var n=t.sibling;if(n!==null){n.return=t.return,B=n;break}B=t.return}}function Kc(e){for(;B!==null;){var t=B;try{switch(t.tag){case 0:case 11:case 15:var n=t.return;try{Ya(4,t)}catch(l){Te(t,n,l)}break;case 1:var r=t.stateNode;if(typeof r.componentDidMount=="function"){var i=t.return;try{r.componentDidMount()}catch(l){Te(t,i,l)}}var o=t.return;try{Zl(t)}catch(l){Te(t,o,l)}break;case 5:var a=t.return;try{Zl(t)}catch(l){Te(t,a,l)}}}catch(l){Te(t,t.return,l)}if(t===e){B=null;break}var s=t.sibling;if(s!==null){s.return=t.return,B=s;break}B=t.return}}var Oy=Math.ceil,Ea=un.ReactCurrentDispatcher,ud=un.ReactCurrentOwner,Ct=un.ReactCurrentBatchConfig,ae=0,Fe=null,Ee=null,Me=0,dt=0,Er=qn(0),Ne=0,$i=null,rr=0,$a=0,dd=0,Ei=null,rt=null,cd=0,Qr=1/0,Jt=null,Pa=!1,tu=null,An=null,jo=!1,wn=null,Na=0,Pi=0,nu=null,Jo=-1,ea=0;function Ge(){return ae&6?Se():Jo!==-1?Jo:Jo=Se()}function Ln(e){return e.mode&1?ae&2&&Me!==0?Me&-Me:wy.transition!==null?(ea===0&&(ea=dm()),ea):(e=de,e!==0||(e=window.event,e=e===void 0?16:_m(e.type)),e):1}function qt(e,t,n,r){if(50<Pi)throw Pi=0,nu=null,Error(L(185));io(e,n,r),(!(ae&2)||e!==Fe)&&(e===Fe&&(!(ae&2)&&($a|=n),Ne===4&&vn(e,Me)),st(e,r),n===1&&ae===0&&!(t.mode&1)&&(Qr=Se()+500,Qa&&On()))}function st(e,t){var n=e.callbackNode;w_(e,t);var r=ha(e,e===Fe?Me:0);if(r===0)n!==null&&rc(n),e.callbackNode=null,e.callbackPriority=0;else if(t=r&-r,e.callbackPriority!==t){if(n!=null&&rc(n),t===1)e.tag===0?ky(Yc.bind(null,e)):Im(Yc.bind(null,e)),yy(function(){!(ae&6)&&On()}),n=null;else{switch(cm(r)){case 1:n=Du;break;case 4:n=lm;break;case 16:n=pa;break;case 536870912:n=um;break;default:n=pa}n=Ff(n,Ef.bind(null,e))}e.callbackPriority=t,e.callbackNode=n}}function Ef(e,t){if(Jo=-1,ea=0,ae&6)throw Error(L(327));var n=e.callbackNode;if(Fr()&&e.callbackNode!==n)return null;var r=ha(e,e===Fe?Me:0);if(r===0)return null;if(r&30||r&e.expiredLanes||t)t=Aa(e,r);else{t=r;var i=ae;ae|=2;var o=Nf();(Fe!==e||Me!==t)&&(Jt=null,Qr=Se()+500,Zn(e,t));do try{By();break}catch(s){Pf(e,s)}while(!0);$u(),Ea.current=o,ae=i,Ee!==null?t=0:(Fe=null,Me=0,t=Ne)}if(t!==0){if(t===2&&(i=Pl(e),i!==0&&(r=i,t=ru(e,i))),t===1)throw n=$i,Zn(e,0),vn(e,r),st(e,Se()),n;if(t===6)vn(e,r);else{if(i=e.current.alternate,!(r&30)&&!Wy(i)&&(t=Aa(e,r),t===2&&(o=Pl(e),o!==0&&(r=o,t=ru(e,o))),t===1))throw n=$i,Zn(e,0),vn(e,r),st(e,Se()),n;switch(e.finishedWork=i,e.finishedLanes=r,t){case 0:case 1:throw Error(L(345));case 2:Hn(e,rt,Jt);break;case 3:if(vn(e,r),(r&130023424)===r&&(t=cd+500-Se(),10<t)){if(ha(e,0)!==0)break;if(i=e.suspendedLanes,(i&r)!==r){Ge(),e.pingedLanes|=e.suspendedLanes&i;break}e.timeoutHandle=Dl(Hn.bind(null,e,rt,Jt),t);break}Hn(e,rt,Jt);break;case 4:if(vn(e,r),(r&4194240)===r)break;for(t=e.eventTimes,i=-1;0<r;){var a=31-Mt(r);o=1<<a,a=t[a],a>i&&(i=a),r&=~o}if(r=i,r=Se()-r,r=(120>r?120:480>r?480:1080>r?1080:1920>r?1920:3e3>r?3e3:4320>r?4320:1960*Oy(r/1960))-r,10<r){e.timeoutHandle=Dl(Hn.bind(null,e,rt,Jt),r);break}Hn(e,rt,Jt);break;case 5:Hn(e,rt,Jt);break;default:throw Error(L(329))}}}return st(e,Se()),e.callbackNode===n?Ef.bind(null,e):null}function ru(e,t){var n=Ei;return e.current.memoizedState.isDehydrated&&(Zn(e,t).flags|=256),e=Aa(e,t),e!==2&&(t=rt,rt=n,t!==null&&iu(t)),e}function iu(e){rt===null?rt=e:rt.push.apply(rt,e)}function Wy(e){for(var t=e;;){if(t.flags&16384){var n=t.updateQueue;if(n!==null&&(n=n.stores,n!==null))for(var r=0;r<n.length;r++){var i=n[r],o=i.getSnapshot;i=i.value;try{if(!Wt(o(),i))return!1}catch{return!1}}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function vn(e,t){for(t&=~dd,t&=~$a,e.suspendedLanes|=t,e.pingedLanes&=~t,e=e.expirationTimes;0<t;){var n=31-Mt(t),r=1<<n;e[n]=-1,t&=~r}}function Yc(e){if(ae&6)throw Error(L(327));Fr();var t=ha(e,0);if(!(t&1))return st(e,Se()),null;var n=Aa(e,t);if(e.tag!==0&&n===2){var r=Pl(e);r!==0&&(t=r,n=ru(e,r))}if(n===1)throw n=$i,Zn(e,0),vn(e,t),st(e,Se()),n;if(n===6)throw Error(L(345));return e.finishedWork=e.current.alternate,e.finishedLanes=t,Hn(e,rt,Jt),st(e,Se()),null}function pd(e,t){var n=ae;ae|=1;try{return e(t)}finally{ae=n,ae===0&&(Qr=Se()+500,Qa&&On())}}function ir(e){wn!==null&&wn.tag===0&&!(ae&6)&&Fr();var t=ae;ae|=1;var n=Ct.transition,r=de;try{if(Ct.transition=null,de=1,e)return e()}finally{de=r,Ct.transition=n,ae=t,!(ae&6)&&On()}}function hd(){dt=Er.current,_e(Er)}function Zn(e,t){e.finishedWork=null,e.finishedLanes=0;var n=e.timeoutHandle;if(n!==-1&&(e.timeoutHandle=-1,_y(n)),Ee!==null)for(n=Ee.return;n!==null;){var r=n;switch(Xu(r),r.tag){case 1:r=r.type.childContextTypes,r!=null&&ya();break;case 3:Ur(),_e(ot),_e(He),nd();break;case 5:td(r);break;case 4:Ur();break;case 13:_e(be);break;case 19:_e(be);break;case 10:Gu(r.type._context);break;case 22:case 23:hd()}n=n.return}if(Fe=e,Ee=e=jn(e.current,null),Me=dt=t,Ne=0,$i=null,dd=$a=rr=0,rt=Ei=null,Yn!==null){for(t=0;t<Yn.length;t++)if(n=Yn[t],r=n.interleaved,r!==null){n.interleaved=null;var i=r.next,o=n.pending;if(o!==null){var a=o.next;o.next=i,r.next=a}n.pending=r}Yn=null}return e}function Pf(e,t){do{var n=Ee;try{if($u(),$o.current=za,Sa){for(var r=ve.memoizedState;r!==null;){var i=r.queue;i!==null&&(i.pending=null),r=r.next}Sa=!1}if(nr=0,je=Pe=ve=null,Si=!1,Xi=0,ud.current=null,n===null||n.return===null){Ne=1,$i=t,Ee=null;break}e:{var o=e,a=n.return,s=n,l=t;if(t=Me,s.flags|=32768,l!==null&&typeof l=="object"&&typeof l.then=="function"){var u=l,c=s,d=c.tag;if(!(c.mode&1)&&(d===0||d===11||d===15)){var h=c.alternate;h?(c.updateQueue=h.updateQueue,c.memoizedState=h.memoizedState,c.lanes=h.lanes):(c.updateQueue=null,c.memoizedState=null)}var m=Ic(a);if(m!==null){m.flags&=-257,Dc(m,a,s,o,t),m.mode&1&&Fc(o,u,t),t=m,l=u;var f=t.updateQueue;if(f===null){var x=new Set;x.add(l),t.updateQueue=x}else f.add(l);break e}else{if(!(t&1)){Fc(o,u,t),md();break e}l=Error(L(426))}}else if(ye&&s.mode&1){var S=Ic(a);if(S!==null){!(S.flags&65536)&&(S.flags|=256),Dc(S,a,s,o,t),Ku(Hr(l,s));break e}}o=l=Hr(l,s),Ne!==4&&(Ne=2),Ei===null?Ei=[o]:Ei.push(o),o=a;do{switch(o.tag){case 3:o.flags|=65536,t&=-t,o.lanes|=t;var g=pf(o,l,t);Pc(o,g);break e;case 1:s=l;var y=o.type,_=o.stateNode;if(!(o.flags&128)&&(typeof y.getDerivedStateFromError=="function"||_!==null&&typeof _.componentDidCatch=="function"&&(An===null||!An.has(_)))){o.flags|=65536,t&=-t,o.lanes|=t;var w=hf(o,s,t);Pc(o,w);break e}}o=o.return}while(o!==null)}Lf(n)}catch(z){t=z,Ee===n&&n!==null&&(Ee=n=n.return);continue}break}while(!0)}function Nf(){var e=Ea.current;return Ea.current=za,e===null?za:e}function md(){(Ne===0||Ne===3||Ne===2)&&(Ne=4),Fe===null||!(rr&268435455)&&!($a&268435455)||vn(Fe,Me)}function Aa(e,t){var n=ae;ae|=2;var r=Nf();(Fe!==e||Me!==t)&&(Jt=null,Zn(e,t));do try{Vy();break}catch(i){Pf(e,i)}while(!0);if($u(),ae=n,Ea.current=r,Ee!==null)throw Error(L(261));return Fe=null,Me=0,Ne}function Vy(){for(;Ee!==null;)Af(Ee)}function By(){for(;Ee!==null&&!m_();)Af(Ee)}function Af(e){var t=Rf(e.alternate,e,dt);e.memoizedProps=e.pendingProps,t===null?Lf(e):Ee=t,ud.current=null}function Lf(e){var t=e;do{var n=t.alternate;if(e=t.return,t.flags&32768){if(n=Iy(n,t),n!==null){n.flags&=32767,Ee=n;return}if(e!==null)e.flags|=32768,e.subtreeFlags=0,e.deletions=null;else{Ne=6,Ee=null;return}}else if(n=Fy(n,t,dt),n!==null){Ee=n;return}if(t=t.sibling,t!==null){Ee=t;return}Ee=t=e}while(t!==null);Ne===0&&(Ne=5)}function Hn(e,t,n){var r=de,i=Ct.transition;try{Ct.transition=null,de=1,Uy(e,t,n,r)}finally{Ct.transition=i,de=r}return null}function Uy(e,t,n,r){do Fr();while(wn!==null);if(ae&6)throw Error(L(327));n=e.finishedWork;var i=e.finishedLanes;if(n===null)return null;if(e.finishedWork=null,e.finishedLanes=0,n===e.current)throw Error(L(177));e.callbackNode=null,e.callbackPriority=0;var o=n.lanes|n.childLanes;if(T_(e,o),e===Fe&&(Ee=Fe=null,Me=0),!(n.subtreeFlags&2064)&&!(n.flags&2064)||jo||(jo=!0,Ff(pa,function(){return Fr(),null})),o=(n.flags&15990)!==0,n.subtreeFlags&15990||o){o=Ct.transition,Ct.transition=null;var a=de;de=1;var s=ae;ae|=4,ud.current=null,My(e,n),Sf(n,e),dy(Fl),ma=!!Rl,Fl=Rl=null,e.current=n,qy(n),f_(),ae=s,de=a,Ct.transition=o}else e.current=n;if(jo&&(jo=!1,wn=e,Na=i),o=e.pendingLanes,o===0&&(An=null),y_(n.stateNode),st(e,Se()),t!==null)for(r=e.onRecoverableError,n=0;n<t.length;n++)i=t[n],r(i.value,{componentStack:i.stack,digest:i.digest});if(Pa)throw Pa=!1,e=tu,tu=null,e;return Na&1&&e.tag!==0&&Fr(),o=e.pendingLanes,o&1?e===nu?Pi++:(Pi=0,nu=e):Pi=0,On(),null}function Fr(){if(wn!==null){var e=cm(Na),t=Ct.transition,n=de;try{if(Ct.transition=null,de=16>e?16:e,wn===null)var r=!1;else{if(e=wn,wn=null,Na=0,ae&6)throw Error(L(331));var i=ae;for(ae|=4,B=e.current;B!==null;){var o=B,a=o.child;if(B.flags&16){var s=o.deletions;if(s!==null){for(var l=0;l<s.length;l++){var u=s[l];for(B=u;B!==null;){var c=B;switch(c.tag){case 0:case 11:case 15:zi(8,c,o)}var d=c.child;if(d!==null)d.return=c,B=d;else for(;B!==null;){c=B;var h=c.sibling,m=c.return;if(wf(c),c===u){B=null;break}if(h!==null){h.return=m,B=h;break}B=m}}}var f=o.alternate;if(f!==null){var x=f.child;if(x!==null){f.child=null;do{var S=x.sibling;x.sibling=null,x=S}while(x!==null)}}B=o}}if(o.subtreeFlags&2064&&a!==null)a.return=o,B=a;else e:for(;B!==null;){if(o=B,o.flags&2048)switch(o.tag){case 0:case 11:case 15:zi(9,o,o.return)}var g=o.sibling;if(g!==null){g.return=o.return,B=g;break e}B=o.return}}var y=e.current;for(B=y;B!==null;){a=B;var _=a.child;if(a.subtreeFlags&2064&&_!==null)_.return=a,B=_;else e:for(a=y;B!==null;){if(s=B,s.flags&2048)try{switch(s.tag){case 0:case 11:case 15:Ya(9,s)}}catch(z){Te(s,s.return,z)}if(s===a){B=null;break e}var w=s.sibling;if(w!==null){w.return=s.return,B=w;break e}B=s.return}}if(ae=i,On(),Xt&&typeof Xt.onPostCommitFiberRoot=="function")try{Xt.onPostCommitFiberRoot(Wa,e)}catch{}r=!0}return r}finally{de=n,Ct.transition=t}}return!1}function $c(e,t,n){t=Hr(n,t),t=pf(e,t,1),e=Nn(e,t,1),t=Ge(),e!==null&&(io(e,1,t),st(e,t))}function Te(e,t,n){if(e.tag===3)$c(e,e,n);else for(;t!==null;){if(t.tag===3){$c(t,e,n);break}else if(t.tag===1){var r=t.stateNode;if(typeof t.type.getDerivedStateFromError=="function"||typeof r.componentDidCatch=="function"&&(An===null||!An.has(r))){e=Hr(n,e),e=hf(t,e,1),t=Nn(t,e,1),e=Ge(),t!==null&&(io(t,1,e),st(t,e));break}}t=t.return}}function Hy(e,t,n){var r=e.pingCache;r!==null&&r.delete(t),t=Ge(),e.pingedLanes|=e.suspendedLanes&n,Fe===e&&(Me&n)===n&&(Ne===4||Ne===3&&(Me&130023424)===Me&&500>Se()-cd?Zn(e,0):dd|=n),st(e,t)}function jf(e,t){t===0&&(e.mode&1?(t=wo,wo<<=1,!(wo&130023424)&&(wo=4194304)):t=1);var n=Ge();e=sn(e,t),e!==null&&(io(e,t,n),st(e,n))}function Qy(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),jf(e,n)}function Xy(e,t){var n=0;switch(e.tag){case 13:var r=e.stateNode,i=e.memoizedState;i!==null&&(n=i.retryLane);break;case 19:r=e.stateNode;break;default:throw Error(L(314))}r!==null&&r.delete(t),jf(e,n)}var Rf;Rf=function(e,t,n){if(e!==null)if(e.memoizedProps!==t.pendingProps||ot.current)it=!0;else{if(!(e.lanes&n)&&!(t.flags&128))return it=!1,Ry(e,t,n);it=!!(e.flags&131072)}else it=!1,ye&&t.flags&1048576&&Dm(t,va,t.index);switch(t.lanes=0,t.tag){case 2:var r=t.type;Zo(e,t),e=t.pendingProps;var i=Wr(t,He.current);Rr(t,n),i=id(null,t,r,e,i,n);var o=od();return t.flags|=1,typeof i=="object"&&i!==null&&typeof i.render=="function"&&i.$$typeof===void 0?(t.tag=1,t.memoizedState=null,t.updateQueue=null,at(r)?(o=!0,xa(t)):o=!1,t.memoizedState=i.state!==null&&i.state!==void 0?i.state:null,Ju(t),i.updater=Ka,t.stateNode=i,i._reactInternals=t,Ul(t,r,e,n),t=Xl(null,t,r,!0,o,n)):(t.tag=0,ye&&o&&Qu(t),Ke(null,t,i,n),t=t.child),t;case 16:r=t.elementType;e:{switch(Zo(e,t),e=t.pendingProps,i=r._init,r=i(r._payload),t.type=r,i=t.tag=Yy(r),e=Rt(r,e),i){case 0:t=Ql(null,t,r,e,n);break e;case 1:t=Oc(null,t,r,e,n);break e;case 11:t=Mc(null,t,r,e,n);break e;case 14:t=qc(null,t,r,Rt(r.type,e),n);break e}throw Error(L(306,r,""))}return t;case 0:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Rt(r,i),Ql(e,t,r,i,n);case 1:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Rt(r,i),Oc(e,t,r,i,n);case 3:e:{if(_f(t),e===null)throw Error(L(387));r=t.pendingProps,o=t.memoizedState,i=o.element,Bm(e,t),Ta(t,r,null,n);var a=t.memoizedState;if(r=a.element,o.isDehydrated)if(o={element:r,isDehydrated:!1,cache:a.cache,pendingSuspenseBoundaries:a.pendingSuspenseBoundaries,transitions:a.transitions},t.updateQueue.baseState=o,t.memoizedState=o,t.flags&256){i=Hr(Error(L(423)),t),t=Wc(e,t,r,n,i);break e}else if(r!==i){i=Hr(Error(L(424)),t),t=Wc(e,t,r,n,i);break e}else for(pt=Pn(t.stateNode.containerInfo.firstChild),ht=t,ye=!0,It=null,n=Wm(t,null,r,n),t.child=n;n;)n.flags=n.flags&-3|4096,n=n.sibling;else{if(Vr(),r===i){t=ln(e,t,n);break e}Ke(e,t,r,n)}t=t.child}return t;case 5:return Um(t),e===null&&Wl(t),r=t.type,i=t.pendingProps,o=e!==null?e.memoizedProps:null,a=i.children,Il(r,i)?a=null:o!==null&&Il(r,o)&&(t.flags|=32),gf(e,t),Ke(e,t,a,n),t.child;case 6:return e===null&&Wl(t),null;case 13:return yf(e,t,n);case 4:return ed(t,t.stateNode.containerInfo),r=t.pendingProps,e===null?t.child=Br(t,null,r,n):Ke(e,t,r,n),t.child;case 11:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Rt(r,i),Mc(e,t,r,i,n);case 7:return Ke(e,t,t.pendingProps,n),t.child;case 8:return Ke(e,t,t.pendingProps.children,n),t.child;case 12:return Ke(e,t,t.pendingProps.children,n),t.child;case 10:e:{if(r=t.type._context,i=t.pendingProps,o=t.memoizedProps,a=i.value,me(ka,r._currentValue),r._currentValue=a,o!==null)if(Wt(o.value,a)){if(o.children===i.children&&!ot.current){t=ln(e,t,n);break e}}else for(o=t.child,o!==null&&(o.return=t);o!==null;){var s=o.dependencies;if(s!==null){a=o.child;for(var l=s.firstContext;l!==null;){if(l.context===r){if(o.tag===1){l=rn(-1,n&-n),l.tag=2;var u=o.updateQueue;if(u!==null){u=u.shared;var c=u.pending;c===null?l.next=l:(l.next=c.next,c.next=l),u.pending=l}}o.lanes|=n,l=o.alternate,l!==null&&(l.lanes|=n),Vl(o.return,n,t),s.lanes|=n;break}l=l.next}}else if(o.tag===10)a=o.type===t.type?null:o.child;else if(o.tag===18){if(a=o.return,a===null)throw Error(L(341));a.lanes|=n,s=a.alternate,s!==null&&(s.lanes|=n),Vl(a,n,t),a=o.sibling}else a=o.child;if(a!==null)a.return=o;else for(a=o;a!==null;){if(a===t){a=null;break}if(o=a.sibling,o!==null){o.return=a.return,a=o;break}a=a.return}o=a}Ke(e,t,i.children,n),t=t.child}return t;case 9:return i=t.type,r=t.pendingProps.children,Rr(t,n),i=St(i),r=r(i),t.flags|=1,Ke(e,t,r,n),t.child;case 14:return r=t.type,i=Rt(r,t.pendingProps),i=Rt(r.type,i),qc(e,t,r,i,n);case 15:return mf(e,t,t.type,t.pendingProps,n);case 17:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Rt(r,i),Zo(e,t),t.tag=1,at(r)?(e=!0,xa(t)):e=!1,Rr(t,n),cf(t,r,i),Ul(t,r,i,n),Xl(null,t,r,!0,e,n);case 19:return xf(e,t,n);case 22:return ff(e,t,n)}throw Error(L(156,t.tag))};function Ff(e,t){return sm(e,t)}function Ky(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function Tt(e,t,n,r){return new Ky(e,t,n,r)}function fd(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Yy(e){if(typeof e=="function")return fd(e)?1:0;if(e!=null){if(e=e.$$typeof,e===Ru)return 11;if(e===Fu)return 14}return 2}function jn(e,t){var n=e.alternate;return n===null?(n=Tt(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&14680064,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n}function ta(e,t,n,r,i,o){var a=2;if(r=e,typeof e=="function")fd(e)&&(a=1);else if(typeof e=="string")a=5;else e:switch(e){case yr:return Jn(n.children,i,o,t);case ju:a=8,i|=8;break;case ml:return e=Tt(12,n,t,i|2),e.elementType=ml,e.lanes=o,e;case fl:return e=Tt(13,n,t,i),e.elementType=fl,e.lanes=o,e;case gl:return e=Tt(19,n,t,i),e.elementType=gl,e.lanes=o,e;case Uh:return Ga(n,i,o,t);default:if(typeof e=="object"&&e!==null)switch(e.$$typeof){case Vh:a=10;break e;case Bh:a=9;break e;case Ru:a=11;break e;case Fu:a=14;break e;case yn:a=16,r=null;break e}throw Error(L(130,e==null?e:typeof e,""))}return t=Tt(a,n,t,i),t.elementType=e,t.type=r,t.lanes=o,t}function Jn(e,t,n,r){return e=Tt(7,e,r,t),e.lanes=n,e}function Ga(e,t,n,r){return e=Tt(22,e,r,t),e.elementType=Uh,e.lanes=n,e.stateNode={isHidden:!1},e}function Ms(e,t,n){return e=Tt(6,e,null,t),e.lanes=n,e}function qs(e,t,n){return t=Tt(4,e.children!==null?e.children:[],e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}function $y(e,t,n,r,i){this.tag=t,this.containerInfo=e,this.finishedWork=this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.pendingContext=this.context=null,this.callbackPriority=0,this.eventTimes=xs(0),this.expirationTimes=xs(-1),this.entangledLanes=this.finishedLanes=this.mutableReadLanes=this.expiredLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=xs(0),this.identifierPrefix=r,this.onRecoverableError=i,this.mutableSourceEagerHydrationData=null}function gd(e,t,n,r,i,o,a,s,l){return e=new $y(e,t,n,s,l),t===1?(t=1,o===!0&&(t|=8)):t=0,o=Tt(3,null,null,t),e.current=o,o.stateNode=e,o.memoizedState={element:r,isDehydrated:n,cache:null,transitions:null,pendingSuspenseBoundaries:null},Ju(o),e}function Gy(e,t,n){var r=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:_r,key:r==null?null:""+r,children:e,containerInfo:t,implementation:n}}function If(e){if(!e)return Dn;e=e._reactInternals;e:{if(lr(e)!==e||e.tag!==1)throw Error(L(170));var t=e;do{switch(t.tag){case 3:t=t.stateNode.context;break e;case 1:if(at(t.type)){t=t.stateNode.__reactInternalMemoizedMergedChildContext;break e}}t=t.return}while(t!==null);throw Error(L(171))}if(e.tag===1){var n=e.type;if(at(n))return Fm(e,n,t)}return t}function Df(e,t,n,r,i,o,a,s,l){return e=gd(n,r,!0,e,i,o,a,s,l),e.context=If(null),n=e.current,r=Ge(),i=Ln(n),o=rn(r,i),o.callback=t??null,Nn(n,o,i),e.current.lanes=i,io(e,i,r),st(e,r),e}function Za(e,t,n,r){var i=t.current,o=Ge(),a=Ln(i);return n=If(n),t.context===null?t.context=n:t.pendingContext=n,t=rn(o,a),t.payload={element:e},r=r===void 0?null:r,r!==null&&(t.callback=r),e=Nn(i,t,a),e!==null&&(qt(e,i,a,o),Yo(e,i,a)),a}function La(e){if(e=e.current,!e.child)return null;switch(e.child.tag){case 5:return e.child.stateNode;default:return e.child.stateNode}}function Gc(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function _d(e,t){Gc(e,t),(e=e.alternate)&&Gc(e,t)}function Zy(){return null}var Mf=typeof reportError=="function"?reportError:function(e){console.error(e)};function yd(e){this._internalRoot=e}Ja.prototype.render=yd.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(L(409));Za(e,t,null,null)};Ja.prototype.unmount=yd.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;ir(function(){Za(null,e,null,null)}),t[an]=null}};function Ja(e){this._internalRoot=e}Ja.prototype.unstable_scheduleHydration=function(e){if(e){var t=mm();e={blockedOn:null,target:e,priority:t};for(var n=0;n<bn.length&&t!==0&&t<bn[n].priority;n++);bn.splice(n,0,e),n===0&&gm(e)}};function xd(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function es(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11&&(e.nodeType!==8||e.nodeValue!==" react-mount-point-unstable "))}function Zc(){}function Jy(e,t,n,r,i){if(i){if(typeof r=="function"){var o=r;r=function(){var u=La(a);o.call(u)}}var a=Df(t,r,e,0,null,!1,!1,"",Zc);return e._reactRootContainer=a,e[an]=a.current,Vi(e.nodeType===8?e.parentNode:e),ir(),a}for(;i=e.lastChild;)e.removeChild(i);if(typeof r=="function"){var s=r;r=function(){var u=La(l);s.call(u)}}var l=gd(e,0,!1,null,null,!1,!1,"",Zc);return e._reactRootContainer=l,e[an]=l.current,Vi(e.nodeType===8?e.parentNode:e),ir(function(){Za(t,l,n,r)}),l}function ts(e,t,n,r,i){var o=n._reactRootContainer;if(o){var a=o;if(typeof i=="function"){var s=i;i=function(){var l=La(a);s.call(l)}}Za(t,a,e,i)}else a=Jy(n,t,e,i,r);return La(a)}pm=function(e){switch(e.tag){case 3:var t=e.stateNode;if(t.current.memoizedState.isDehydrated){var n=yi(t.pendingLanes);n!==0&&(Mu(t,n|1),st(t,Se()),!(ae&6)&&(Qr=Se()+500,On()))}break;case 13:ir(function(){var r=sn(e,1);if(r!==null){var i=Ge();qt(r,e,1,i)}}),_d(e,1)}};qu=function(e){if(e.tag===13){var t=sn(e,134217728);if(t!==null){var n=Ge();qt(t,e,134217728,n)}_d(e,134217728)}};hm=function(e){if(e.tag===13){var t=Ln(e),n=sn(e,t);if(n!==null){var r=Ge();qt(n,e,t,r)}_d(e,t)}};mm=function(){return de};fm=function(e,t){var n=de;try{return de=e,t()}finally{de=n}};Sl=function(e,t,n){switch(t){case"input":if(xl(e,n),t=n.name,n.type==="radio"&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll("input[name="+JSON.stringify(""+t)+'][type="radio"]'),t=0;t<n.length;t++){var r=n[t];if(r!==e&&r.form===e.form){var i=Ha(r);if(!i)throw Error(L(90));Qh(r),xl(r,i)}}}break;case"textarea":Kh(e,n);break;case"select":t=n.value,t!=null&&Nr(e,!!n.multiple,t,!1)}};tm=pd;nm=ir;var e1={usingClientEntryPoint:!1,Events:[ao,kr,Ha,Jh,em,pd]},di={findFiberByHostInstance:Kn,bundleType:0,version:"18.3.1",rendererPackageName:"react-dom"},t1={bundleType:di.bundleType,version:di.version,rendererPackageName:di.rendererPackageName,rendererConfig:di.rendererConfig,overrideHookState:null,overrideHookStateDeletePath:null,overrideHookStateRenamePath:null,overrideProps:null,overridePropsDeletePath:null,overridePropsRenamePath:null,setErrorHandler:null,setSuspenseHandler:null,scheduleUpdate:null,currentDispatcherRef:un.ReactCurrentDispatcher,findHostInstanceByFiber:function(e){return e=om(e),e===null?null:e.stateNode},findFiberByHostInstance:di.findFiberByHostInstance||Zy,findHostInstancesForRefresh:null,scheduleRefresh:null,scheduleRoot:null,setRefreshHandler:null,getCurrentFiber:null,reconcilerVersion:"18.3.1-next-f1338f8080-20240426"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"){var Ro=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Ro.isDisabled&&Ro.supportsFiber)try{Wa=Ro.inject(t1),Xt=Ro}catch{}}gt.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=e1;gt.createPortal=function(e,t){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!xd(t))throw Error(L(200));return Gy(e,t,null,n)};gt.createRoot=function(e,t){if(!xd(e))throw Error(L(299));var n=!1,r="",i=Mf;return t!=null&&(t.unstable_strictMode===!0&&(n=!0),t.identifierPrefix!==void 0&&(r=t.identifierPrefix),t.onRecoverableError!==void 0&&(i=t.onRecoverableError)),t=gd(e,1,!1,null,null,n,!1,r,i),e[an]=t.current,Vi(e.nodeType===8?e.parentNode:e),new yd(t)};gt.findDOMNode=function(e){if(e==null)return null;if(e.nodeType===1)return e;var t=e._reactInternals;if(t===void 0)throw typeof e.render=="function"?Error(L(188)):(e=Object.keys(e).join(","),Error(L(268,e)));return e=om(t),e=e===null?null:e.stateNode,e};gt.flushSync=function(e){return ir(e)};gt.hydrate=function(e,t,n){if(!es(t))throw Error(L(200));return ts(null,e,t,!0,n)};gt.hydrateRoot=function(e,t,n){if(!xd(e))throw Error(L(405));var r=n!=null&&n.hydratedSources||null,i=!1,o="",a=Mf;if(n!=null&&(n.unstable_strictMode===!0&&(i=!0),n.identifierPrefix!==void 0&&(o=n.identifierPrefix),n.onRecoverableError!==void 0&&(a=n.onRecoverableError)),t=Df(t,null,e,1,n??null,i,!1,o,a),e[an]=t.current,Vi(e),r)for(e=0;e<r.length;e++)n=r[e],i=n._getVersion,i=i(n._source),t.mutableSourceEagerHydrationData==null?t.mutableSourceEagerHydrationData=[n,i]:t.mutableSourceEagerHydrationData.push(n,i);return new Ja(t)};gt.render=function(e,t,n){if(!es(t))throw Error(L(200));return ts(null,e,t,!1,n)};gt.unmountComponentAtNode=function(e){if(!es(e))throw Error(L(40));return e._reactRootContainer?(ir(function(){ts(null,null,e,!1,function(){e._reactRootContainer=null,e[an]=null})}),!0):!1};gt.unstable_batchedUpdates=pd;gt.unstable_renderSubtreeIntoContainer=function(e,t,n,r){if(!es(n))throw Error(L(200));if(e==null||e._reactInternals===void 0)throw Error(L(38));return ts(e,t,n,!1,r)};gt.version="18.3.1-next-f1338f8080-20240426";function qf(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(qf)}catch(e){console.error(e)}}qf(),Mh.exports=gt;var n1=Mh.exports,Jc=n1;pl.createRoot=Jc.createRoot,pl.hydrateRoot=Jc.hydrateRoot;/**
 * @remix-run/router v1.23.2
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Gi(){return Gi=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Gi.apply(this,arguments)}var Tn;(function(e){e.Pop="POP",e.Push="PUSH",e.Replace="REPLACE"})(Tn||(Tn={}));const ep="popstate";function r1(e){e===void 0&&(e={});function t(r,i){let{pathname:o,search:a,hash:s}=r.location;return ou("",{pathname:o,search:a,hash:s},i.state&&i.state.usr||null,i.state&&i.state.key||"default")}function n(r,i){return typeof i=="string"?i:ja(i)}return o1(t,n,null,e)}function ke(e,t){if(e===!1||e===null||typeof e>"u")throw new Error(t)}function bd(e,t){if(!e){typeof console<"u"&&console.warn(t);try{throw new Error(t)}catch{}}}function i1(){return Math.random().toString(36).substr(2,8)}function tp(e,t){return{usr:e.state,key:e.key,idx:t}}function ou(e,t,n,r){return n===void 0&&(n=null),Gi({pathname:typeof e=="string"?e:e.pathname,search:"",hash:""},typeof t=="string"?Zr(t):t,{state:n,key:t&&t.key||r||i1()})}function ja(e){let{pathname:t="/",search:n="",hash:r=""}=e;return n&&n!=="?"&&(t+=n.charAt(0)==="?"?n:"?"+n),r&&r!=="#"&&(t+=r.charAt(0)==="#"?r:"#"+r),t}function Zr(e){let t={};if(e){let n=e.indexOf("#");n>=0&&(t.hash=e.substr(n),e=e.substr(0,n));let r=e.indexOf("?");r>=0&&(t.search=e.substr(r),e=e.substr(0,r)),e&&(t.pathname=e)}return t}function o1(e,t,n,r){r===void 0&&(r={});let{window:i=document.defaultView,v5Compat:o=!1}=r,a=i.history,s=Tn.Pop,l=null,u=c();u==null&&(u=0,a.replaceState(Gi({},a.state,{idx:u}),""));function c(){return(a.state||{idx:null}).idx}function d(){s=Tn.Pop;let S=c(),g=S==null?null:S-u;u=S,l&&l({action:s,location:x.location,delta:g})}function h(S,g){s=Tn.Push;let y=ou(x.location,S,g);u=c()+1;let _=tp(y,u),w=x.createHref(y);try{a.pushState(_,"",w)}catch(z){if(z instanceof DOMException&&z.name==="DataCloneError")throw z;i.location.assign(w)}o&&l&&l({action:s,location:x.location,delta:1})}function m(S,g){s=Tn.Replace;let y=ou(x.location,S,g);u=c();let _=tp(y,u),w=x.createHref(y);a.replaceState(_,"",w),o&&l&&l({action:s,location:x.location,delta:0})}function f(S){let g=i.location.origin!=="null"?i.location.origin:i.location.href,y=typeof S=="string"?S:ja(S);return y=y.replace(/ $/,"%20"),ke(g,"No window.location.(origin|href) available to create URL for href: "+y),new URL(y,g)}let x={get action(){return s},get location(){return e(i,a)},listen(S){if(l)throw new Error("A history only accepts one active listener");return i.addEventListener(ep,d),l=S,()=>{i.removeEventListener(ep,d),l=null}},createHref(S){return t(i,S)},createURL:f,encodeLocation(S){let g=f(S);return{pathname:g.pathname,search:g.search,hash:g.hash}},push:h,replace:m,go(S){return a.go(S)}};return x}var np;(function(e){e.data="data",e.deferred="deferred",e.redirect="redirect",e.error="error"})(np||(np={}));function a1(e,t,n){return n===void 0&&(n="/"),s1(e,t,n)}function s1(e,t,n,r){let i=typeof t=="string"?Zr(t):t,o=Xr(i.pathname||"/",n);if(o==null)return null;let a=Of(e);l1(a);let s=null;for(let l=0;s==null&&l<a.length;++l){let u=x1(o);s=_1(a[l],u)}return s}function Of(e,t,n,r){t===void 0&&(t=[]),n===void 0&&(n=[]),r===void 0&&(r="");let i=(o,a,s)=>{let l={relativePath:s===void 0?o.path||"":s,caseSensitive:o.caseSensitive===!0,childrenIndex:a,route:o};l.relativePath.startsWith("/")&&(ke(l.relativePath.startsWith(r),'Absolute route path "'+l.relativePath+'" nested under path '+('"'+r+'" is not valid. An absolute child route path ')+"must start with the combined path of all its parent routes."),l.relativePath=l.relativePath.slice(r.length));let u=Rn([r,l.relativePath]),c=n.concat(l);o.children&&o.children.length>0&&(ke(o.index!==!0,"Index routes must not have child routes. Please remove "+('all child routes from route path "'+u+'".')),Of(o.children,t,c,u)),!(o.path==null&&!o.index)&&t.push({path:u,score:f1(u,o.index),routesMeta:c})};return e.forEach((o,a)=>{var s;if(o.path===""||!((s=o.path)!=null&&s.includes("?")))i(o,a);else for(let l of Wf(o.path))i(o,a,l)}),t}function Wf(e){let t=e.split("/");if(t.length===0)return[];let[n,...r]=t,i=n.endsWith("?"),o=n.replace(/\?$/,"");if(r.length===0)return i?[o,""]:[o];let a=Wf(r.join("/")),s=[];return s.push(...a.map(l=>l===""?o:[o,l].join("/"))),i&&s.push(...a),s.map(l=>e.startsWith("/")&&l===""?"/":l)}function l1(e){e.sort((t,n)=>t.score!==n.score?n.score-t.score:g1(t.routesMeta.map(r=>r.childrenIndex),n.routesMeta.map(r=>r.childrenIndex)))}const u1=/^:[\w-]+$/,d1=3,c1=2,p1=1,h1=10,m1=-2,rp=e=>e==="*";function f1(e,t){let n=e.split("/"),r=n.length;return n.some(rp)&&(r+=m1),t&&(r+=c1),n.filter(i=>!rp(i)).reduce((i,o)=>i+(u1.test(o)?d1:o===""?p1:h1),r)}function g1(e,t){return e.length===t.length&&e.slice(0,-1).every((r,i)=>r===t[i])?e[e.length-1]-t[t.length-1]:0}function _1(e,t,n){let{routesMeta:r}=e,i={},o="/",a=[];for(let s=0;s<r.length;++s){let l=r[s],u=s===r.length-1,c=o==="/"?t:t.slice(o.length)||"/",d=au({path:l.relativePath,caseSensitive:l.caseSensitive,end:u},c),h=l.route;if(!d)return null;Object.assign(i,d.params),a.push({params:i,pathname:Rn([o,d.pathname]),pathnameBase:T1(Rn([o,d.pathnameBase])),route:h}),d.pathnameBase!=="/"&&(o=Rn([o,d.pathnameBase]))}return a}function au(e,t){typeof e=="string"&&(e={path:e,caseSensitive:!1,end:!0});let[n,r]=y1(e.path,e.caseSensitive,e.end),i=t.match(n);if(!i)return null;let o=i[0],a=o.replace(/(.)\/+$/,"$1"),s=i.slice(1);return{params:r.reduce((u,c,d)=>{let{paramName:h,isOptional:m}=c;if(h==="*"){let x=s[d]||"";a=o.slice(0,o.length-x.length).replace(/(.)\/+$/,"$1")}const f=s[d];return m&&!f?u[h]=void 0:u[h]=(f||"").replace(/%2F/g,"/"),u},{}),pathname:o,pathnameBase:a,pattern:e}}function y1(e,t,n){t===void 0&&(t=!1),n===void 0&&(n=!0),bd(e==="*"||!e.endsWith("*")||e.endsWith("/*"),'Route path "'+e+'" will be treated as if it were '+('"'+e.replace(/\*$/,"/*")+'" because the `*` character must ')+"always follow a `/` in the pattern. To get rid of this warning, "+('please change the route path to "'+e.replace(/\*$/,"/*")+'".'));let r=[],i="^"+e.replace(/\/*\*?$/,"").replace(/^\/*/,"/").replace(/[\\.*+^${}|()[\]]/g,"\\$&").replace(/\/:([\w-]+)(\?)?/g,(a,s,l)=>(r.push({paramName:s,isOptional:l!=null}),l?"/?([^\\/]+)?":"/([^\\/]+)"));return e.endsWith("*")?(r.push({paramName:"*"}),i+=e==="*"||e==="/*"?"(.*)$":"(?:\\/(.+)|\\/*)$"):n?i+="\\/*$":e!==""&&e!=="/"&&(i+="(?:(?=\\/|$))"),[new RegExp(i,t?void 0:"i"),r]}function x1(e){try{return e.split("/").map(t=>decodeURIComponent(t).replace(/\//g,"%2F")).join("/")}catch(t){return bd(!1,'The URL path "'+e+'" could not be decoded because it is is a malformed URL segment. This is probably due to a bad percent '+("encoding ("+t+").")),e}}function Xr(e,t){if(t==="/")return e;if(!e.toLowerCase().startsWith(t.toLowerCase()))return null;let n=t.endsWith("/")?t.length-1:t.length,r=e.charAt(n);return r&&r!=="/"?null:e.slice(n)||"/"}const b1=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,v1=e=>b1.test(e);function k1(e,t){t===void 0&&(t="/");let{pathname:n,search:r="",hash:i=""}=typeof e=="string"?Zr(e):e,o;if(n)if(v1(n))o=n;else{if(n.includes("//")){let a=n;n=n.replace(/\/\/+/g,"/"),bd(!1,"Pathnames cannot have embedded double slashes - normalizing "+(a+" -> "+n))}n.startsWith("/")?o=ip(n.substring(1),"/"):o=ip(n,t)}else o=t;return{pathname:o,search:C1(r),hash:S1(i)}}function ip(e,t){let n=t.replace(/\/+$/,"").split("/");return e.split("/").forEach(i=>{i===".."?n.length>1&&n.pop():i!=="."&&n.push(i)}),n.length>1?n.join("/"):"/"}function Os(e,t,n,r){return"Cannot include a '"+e+"' character in a manually specified "+("`to."+t+"` field ["+JSON.stringify(r)+"].  Please separate it out to the ")+("`to."+n+"` field. Alternatively you may provide the full path as ")+'a string in <Link to="..."> and the router will parse it for you.'}function w1(e){return e.filter((t,n)=>n===0||t.route.path&&t.route.path.length>0)}function vd(e,t){let n=w1(e);return t?n.map((r,i)=>i===n.length-1?r.pathname:r.pathnameBase):n.map(r=>r.pathnameBase)}function kd(e,t,n,r){r===void 0&&(r=!1);let i;typeof e=="string"?i=Zr(e):(i=Gi({},e),ke(!i.pathname||!i.pathname.includes("?"),Os("?","pathname","search",i)),ke(!i.pathname||!i.pathname.includes("#"),Os("#","pathname","hash",i)),ke(!i.search||!i.search.includes("#"),Os("#","search","hash",i)));let o=e===""||i.pathname==="",a=o?"/":i.pathname,s;if(a==null)s=n;else{let d=t.length-1;if(!r&&a.startsWith("..")){let h=a.split("/");for(;h[0]==="..";)h.shift(),d-=1;i.pathname=h.join("/")}s=d>=0?t[d]:"/"}let l=k1(i,s),u=a&&a!=="/"&&a.endsWith("/"),c=(o||a===".")&&n.endsWith("/");return!l.pathname.endsWith("/")&&(u||c)&&(l.pathname+="/"),l}const Rn=e=>e.join("/").replace(/\/\/+/g,"/"),T1=e=>e.replace(/\/+$/,"").replace(/^\/*/,"/"),C1=e=>!e||e==="?"?"":e.startsWith("?")?e:"?"+e,S1=e=>!e||e==="#"?"":e.startsWith("#")?e:"#"+e;function z1(e){return e!=null&&typeof e.status=="number"&&typeof e.statusText=="string"&&typeof e.internal=="boolean"&&"data"in e}const Vf=["post","put","patch","delete"];new Set(Vf);const E1=["get",...Vf];new Set(E1);/**
 * React Router v6.30.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Zi(){return Zi=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Zi.apply(this,arguments)}const ns=C.createContext(null),Bf=C.createContext(null),dn=C.createContext(null),rs=C.createContext(null),cn=C.createContext({outlet:null,matches:[],isDataRoute:!1}),Uf=C.createContext(null);function P1(e,t){let{relative:n}=t===void 0?{}:t;Jr()||ke(!1);let{basename:r,navigator:i}=C.useContext(dn),{hash:o,pathname:a,search:s}=is(e,{relative:n}),l=a;return r!=="/"&&(l=a==="/"?r:Rn([r,a])),i.createHref({pathname:l,search:s,hash:o})}function Jr(){return C.useContext(rs)!=null}function ur(){return Jr()||ke(!1),C.useContext(rs).location}function Hf(e){C.useContext(dn).static||C.useLayoutEffect(e)}function Qf(){let{isDataRoute:e}=C.useContext(cn);return e?V1():N1()}function N1(){Jr()||ke(!1);let e=C.useContext(ns),{basename:t,future:n,navigator:r}=C.useContext(dn),{matches:i}=C.useContext(cn),{pathname:o}=ur(),a=JSON.stringify(vd(i,n.v7_relativeSplatPath)),s=C.useRef(!1);return Hf(()=>{s.current=!0}),C.useCallback(function(u,c){if(c===void 0&&(c={}),!s.current)return;if(typeof u=="number"){r.go(u);return}let d=kd(u,JSON.parse(a),o,c.relative==="path");e==null&&t!=="/"&&(d.pathname=d.pathname==="/"?t:Rn([t,d.pathname])),(c.replace?r.replace:r.push)(d,c.state,c)},[t,r,a,o,e])}function Xf(){let{matches:e}=C.useContext(cn),t=e[e.length-1];return t?t.params:{}}function is(e,t){let{relative:n}=t===void 0?{}:t,{future:r}=C.useContext(dn),{matches:i}=C.useContext(cn),{pathname:o}=ur(),a=JSON.stringify(vd(i,r.v7_relativeSplatPath));return C.useMemo(()=>kd(e,JSON.parse(a),o,n==="path"),[e,a,o,n])}function A1(e,t){return L1(e,t)}function L1(e,t,n,r){Jr()||ke(!1);let{navigator:i}=C.useContext(dn),{matches:o}=C.useContext(cn),a=o[o.length-1],s=a?a.params:{};a&&a.pathname;let l=a?a.pathnameBase:"/";a&&a.route;let u=ur(),c;if(t){var d;let S=typeof t=="string"?Zr(t):t;l==="/"||(d=S.pathname)!=null&&d.startsWith(l)||ke(!1),c=S}else c=u;let h=c.pathname||"/",m=h;if(l!=="/"){let S=l.replace(/^\//,"").split("/");m="/"+h.replace(/^\//,"").split("/").slice(S.length).join("/")}let f=a1(e,{pathname:m}),x=D1(f&&f.map(S=>Object.assign({},S,{params:Object.assign({},s,S.params),pathname:Rn([l,i.encodeLocation?i.encodeLocation(S.pathname).pathname:S.pathname]),pathnameBase:S.pathnameBase==="/"?l:Rn([l,i.encodeLocation?i.encodeLocation(S.pathnameBase).pathname:S.pathnameBase])})),o,n,r);return t&&x?C.createElement(rs.Provider,{value:{location:Zi({pathname:"/",search:"",hash:"",state:null,key:"default"},c),navigationType:Tn.Pop}},x):x}function j1(){let e=W1(),t=z1(e)?e.status+" "+e.statusText:e instanceof Error?e.message:JSON.stringify(e),n=e instanceof Error?e.stack:null,i={padding:"0.5rem",backgroundColor:"rgba(200,200,200, 0.5)"};return C.createElement(C.Fragment,null,C.createElement("h2",null,"Unexpected Application Error!"),C.createElement("h3",{style:{fontStyle:"italic"}},t),n?C.createElement("pre",{style:i},n):null,null)}const R1=C.createElement(j1,null);class F1 extends C.Component{constructor(t){super(t),this.state={location:t.location,revalidation:t.revalidation,error:t.error}}static getDerivedStateFromError(t){return{error:t}}static getDerivedStateFromProps(t,n){return n.location!==t.location||n.revalidation!=="idle"&&t.revalidation==="idle"?{error:t.error,location:t.location,revalidation:t.revalidation}:{error:t.error!==void 0?t.error:n.error,location:n.location,revalidation:t.revalidation||n.revalidation}}componentDidCatch(t,n){console.error("React Router caught the following error during render",t,n)}render(){return this.state.error!==void 0?C.createElement(cn.Provider,{value:this.props.routeContext},C.createElement(Uf.Provider,{value:this.state.error,children:this.props.component})):this.props.children}}function I1(e){let{routeContext:t,match:n,children:r}=e,i=C.useContext(ns);return i&&i.static&&i.staticContext&&(n.route.errorElement||n.route.ErrorBoundary)&&(i.staticContext._deepestRenderedBoundaryId=n.route.id),C.createElement(cn.Provider,{value:t},r)}function D1(e,t,n,r){var i;if(t===void 0&&(t=[]),n===void 0&&(n=null),r===void 0&&(r=null),e==null){var o;if(!n)return null;if(n.errors)e=n.matches;else if((o=r)!=null&&o.v7_partialHydration&&t.length===0&&!n.initialized&&n.matches.length>0)e=n.matches;else return null}let a=e,s=(i=n)==null?void 0:i.errors;if(s!=null){let c=a.findIndex(d=>d.route.id&&(s==null?void 0:s[d.route.id])!==void 0);c>=0||ke(!1),a=a.slice(0,Math.min(a.length,c+1))}let l=!1,u=-1;if(n&&r&&r.v7_partialHydration)for(let c=0;c<a.length;c++){let d=a[c];if((d.route.HydrateFallback||d.route.hydrateFallbackElement)&&(u=c),d.route.id){let{loaderData:h,errors:m}=n,f=d.route.loader&&h[d.route.id]===void 0&&(!m||m[d.route.id]===void 0);if(d.route.lazy||f){l=!0,u>=0?a=a.slice(0,u+1):a=[a[0]];break}}}return a.reduceRight((c,d,h)=>{let m,f=!1,x=null,S=null;n&&(m=s&&d.route.id?s[d.route.id]:void 0,x=d.route.errorElement||R1,l&&(u<0&&h===0?(B1("route-fallback"),f=!0,S=null):u===h&&(f=!0,S=d.route.hydrateFallbackElement||null)));let g=t.concat(a.slice(0,h+1)),y=()=>{let _;return m?_=x:f?_=S:d.route.Component?_=C.createElement(d.route.Component,null):d.route.element?_=d.route.element:_=c,C.createElement(I1,{match:d,routeContext:{outlet:c,matches:g,isDataRoute:n!=null},children:_})};return n&&(d.route.ErrorBoundary||d.route.errorElement||h===0)?C.createElement(F1,{location:n.location,revalidation:n.revalidation,component:x,error:m,children:y(),routeContext:{outlet:null,matches:g,isDataRoute:!0}}):y()},null)}var Kf=function(e){return e.UseBlocker="useBlocker",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e}(Kf||{}),Yf=function(e){return e.UseBlocker="useBlocker",e.UseLoaderData="useLoaderData",e.UseActionData="useActionData",e.UseRouteError="useRouteError",e.UseNavigation="useNavigation",e.UseRouteLoaderData="useRouteLoaderData",e.UseMatches="useMatches",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e.UseRouteId="useRouteId",e}(Yf||{});function M1(e){let t=C.useContext(ns);return t||ke(!1),t}function q1(e){let t=C.useContext(Bf);return t||ke(!1),t}function O1(e){let t=C.useContext(cn);return t||ke(!1),t}function $f(e){let t=O1(),n=t.matches[t.matches.length-1];return n.route.id||ke(!1),n.route.id}function W1(){var e;let t=C.useContext(Uf),n=q1(),r=$f();return t!==void 0?t:(e=n.errors)==null?void 0:e[r]}function V1(){let{router:e}=M1(Kf.UseNavigateStable),t=$f(Yf.UseNavigateStable),n=C.useRef(!1);return Hf(()=>{n.current=!0}),C.useCallback(function(i,o){o===void 0&&(o={}),n.current&&(typeof i=="number"?e.navigate(i):e.navigate(i,Zi({fromRouteId:t},o)))},[e,t])}const op={};function B1(e,t,n){op[e]||(op[e]=!0)}function U1(e,t){e==null||e.v7_startTransition,e==null||e.v7_relativeSplatPath}function Gf(e){let{to:t,replace:n,state:r,relative:i}=e;Jr()||ke(!1);let{future:o,static:a}=C.useContext(dn),{matches:s}=C.useContext(cn),{pathname:l}=ur(),u=Qf(),c=kd(t,vd(s,o.v7_relativeSplatPath),l,i==="path"),d=JSON.stringify(c);return C.useEffect(()=>u(JSON.parse(d),{replace:n,state:r,relative:i}),[u,d,i,n,r]),null}function Qn(e){ke(!1)}function H1(e){let{basename:t="/",children:n=null,location:r,navigationType:i=Tn.Pop,navigator:o,static:a=!1,future:s}=e;Jr()&&ke(!1);let l=t.replace(/^\/*/,"/"),u=C.useMemo(()=>({basename:l,navigator:o,static:a,future:Zi({v7_relativeSplatPath:!1},s)}),[l,s,o,a]);typeof r=="string"&&(r=Zr(r));let{pathname:c="/",search:d="",hash:h="",state:m=null,key:f="default"}=r,x=C.useMemo(()=>{let S=Xr(c,l);return S==null?null:{location:{pathname:S,search:d,hash:h,state:m,key:f},navigationType:i}},[l,c,d,h,m,f,i]);return x==null?null:C.createElement(dn.Provider,{value:u},C.createElement(rs.Provider,{children:n,value:x}))}function Q1(e){let{children:t,location:n}=e;return A1(su(t),n)}new Promise(()=>{});function su(e,t){t===void 0&&(t=[]);let n=[];return C.Children.forEach(e,(r,i)=>{if(!C.isValidElement(r))return;let o=[...t,i];if(r.type===C.Fragment){n.push.apply(n,su(r.props.children,o));return}r.type!==Qn&&ke(!1),!r.props.index||!r.props.children||ke(!1);let a={id:r.props.id||o.join("-"),caseSensitive:r.props.caseSensitive,element:r.props.element,Component:r.props.Component,index:r.props.index,path:r.props.path,loader:r.props.loader,action:r.props.action,errorElement:r.props.errorElement,ErrorBoundary:r.props.ErrorBoundary,hasErrorBoundary:r.props.ErrorBoundary!=null||r.props.errorElement!=null,shouldRevalidate:r.props.shouldRevalidate,handle:r.props.handle,lazy:r.props.lazy};r.props.children&&(a.children=su(r.props.children,o)),n.push(a)}),n}/**
 * React Router DOM v6.30.3
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Ra(){return Ra=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Ra.apply(this,arguments)}function Zf(e,t){if(e==null)return{};var n={},r=Object.keys(e),i,o;for(o=0;o<r.length;o++)i=r[o],!(t.indexOf(i)>=0)&&(n[i]=e[i]);return n}function X1(e){return!!(e.metaKey||e.altKey||e.ctrlKey||e.shiftKey)}function K1(e,t){return e.button===0&&(!t||t==="_self")&&!X1(e)}const Y1=["onClick","relative","reloadDocument","replace","state","target","to","preventScrollReset","viewTransition"],$1=["aria-current","caseSensitive","className","end","style","to","viewTransition","children"],G1="6";try{window.__reactRouterVersion=G1}catch{}const Z1=C.createContext({isTransitioning:!1}),J1="startTransition",ap=Hg[J1];function ex(e){let{basename:t,children:n,future:r,window:i}=e,o=C.useRef();o.current==null&&(o.current=r1({window:i,v5Compat:!0}));let a=o.current,[s,l]=C.useState({action:a.action,location:a.location}),{v7_startTransition:u}=r||{},c=C.useCallback(d=>{u&&ap?ap(()=>l(d)):l(d)},[l,u]);return C.useLayoutEffect(()=>a.listen(c),[a,c]),C.useEffect(()=>U1(r),[r]),C.createElement(H1,{basename:t,children:n,location:s.location,navigationType:s.action,navigator:a,future:r})}const tx=typeof window<"u"&&typeof window.document<"u"&&typeof window.document.createElement<"u",nx=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,Ye=C.forwardRef(function(t,n){let{onClick:r,relative:i,reloadDocument:o,replace:a,state:s,target:l,to:u,preventScrollReset:c,viewTransition:d}=t,h=Zf(t,Y1),{basename:m}=C.useContext(dn),f,x=!1;if(typeof u=="string"&&nx.test(u)&&(f=u,tx))try{let _=new URL(window.location.href),w=u.startsWith("//")?new URL(_.protocol+u):new URL(u),z=Xr(w.pathname,m);w.origin===_.origin&&z!=null?u=z+w.search+w.hash:x=!0}catch{}let S=P1(u,{relative:i}),g=ox(u,{replace:a,state:s,target:l,preventScrollReset:c,relative:i,viewTransition:d});function y(_){r&&r(_),_.defaultPrevented||g(_)}return C.createElement("a",Ra({},h,{href:f||S,onClick:x||o?r:y,ref:n,target:l}))}),rx=C.forwardRef(function(t,n){let{"aria-current":r="page",caseSensitive:i=!1,className:o="",end:a=!1,style:s,to:l,viewTransition:u,children:c}=t,d=Zf(t,$1),h=is(l,{relative:d.relative}),m=ur(),f=C.useContext(Bf),{navigator:x,basename:S}=C.useContext(dn),g=f!=null&&ax(h)&&u===!0,y=x.encodeLocation?x.encodeLocation(h).pathname:h.pathname,_=m.pathname,w=f&&f.navigation&&f.navigation.location?f.navigation.location.pathname:null;i||(_=_.toLowerCase(),w=w?w.toLowerCase():null,y=y.toLowerCase()),w&&S&&(w=Xr(w,S)||w);const z=y!=="/"&&y.endsWith("/")?y.length-1:y.length;let k=_===y||!a&&_.startsWith(y)&&_.charAt(z)==="/",E=w!=null&&(w===y||!a&&w.startsWith(y)&&w.charAt(y.length)==="/"),N={isActive:k,isPending:E,isTransitioning:g},q=k?r:void 0,T;typeof o=="function"?T=o(N):T=[o,k?"active":null,E?"pending":null,g?"transitioning":null].filter(Boolean).join(" ");let D=typeof s=="function"?s(N):s;return C.createElement(Ye,Ra({},d,{"aria-current":q,className:T,ref:n,style:D,to:l,viewTransition:u}),typeof c=="function"?c(N):c)});var lu;(function(e){e.UseScrollRestoration="useScrollRestoration",e.UseSubmit="useSubmit",e.UseSubmitFetcher="useSubmitFetcher",e.UseFetcher="useFetcher",e.useViewTransitionState="useViewTransitionState"})(lu||(lu={}));var sp;(function(e){e.UseFetcher="useFetcher",e.UseFetchers="useFetchers",e.UseScrollRestoration="useScrollRestoration"})(sp||(sp={}));function ix(e){let t=C.useContext(ns);return t||ke(!1),t}function ox(e,t){let{target:n,replace:r,state:i,preventScrollReset:o,relative:a,viewTransition:s}=t===void 0?{}:t,l=Qf(),u=ur(),c=is(e,{relative:a});return C.useCallback(d=>{if(K1(d,n)){d.preventDefault();let h=r!==void 0?r:ja(u)===ja(c);l(e,{replace:h,state:i,preventScrollReset:o,relative:a,viewTransition:s})}},[u,l,c,r,i,n,e,o,a,s])}function ax(e,t){t===void 0&&(t={});let n=C.useContext(Z1);n==null&&ke(!1);let{basename:r}=ix(lu.useViewTransitionState),i=is(e,{relative:t.relative});if(!n.isTransitioning)return!1;let o=Xr(n.currentLocation.pathname,r)||n.currentLocation.pathname,a=Xr(n.nextLocation.pathname,r)||n.nextLocation.pathname;return au(i.pathname,a)!=null||au(i.pathname,o)!=null}var sx=typeof Element<"u",lx=typeof Map=="function",ux=typeof Set=="function",dx=typeof ArrayBuffer=="function"&&!!ArrayBuffer.isView;function na(e,t){if(e===t)return!0;if(e&&t&&typeof e=="object"&&typeof t=="object"){if(e.constructor!==t.constructor)return!1;var n,r,i;if(Array.isArray(e)){if(n=e.length,n!=t.length)return!1;for(r=n;r--!==0;)if(!na(e[r],t[r]))return!1;return!0}var o;if(lx&&e instanceof Map&&t instanceof Map){if(e.size!==t.size)return!1;for(o=e.entries();!(r=o.next()).done;)if(!t.has(r.value[0]))return!1;for(o=e.entries();!(r=o.next()).done;)if(!na(r.value[1],t.get(r.value[0])))return!1;return!0}if(ux&&e instanceof Set&&t instanceof Set){if(e.size!==t.size)return!1;for(o=e.entries();!(r=o.next()).done;)if(!t.has(r.value[0]))return!1;return!0}if(dx&&ArrayBuffer.isView(e)&&ArrayBuffer.isView(t)){if(n=e.length,n!=t.length)return!1;for(r=n;r--!==0;)if(e[r]!==t[r])return!1;return!0}if(e.constructor===RegExp)return e.source===t.source&&e.flags===t.flags;if(e.valueOf!==Object.prototype.valueOf&&typeof e.valueOf=="function"&&typeof t.valueOf=="function")return e.valueOf()===t.valueOf();if(e.toString!==Object.prototype.toString&&typeof e.toString=="function"&&typeof t.toString=="function")return e.toString()===t.toString();if(i=Object.keys(e),n=i.length,n!==Object.keys(t).length)return!1;for(r=n;r--!==0;)if(!Object.prototype.hasOwnProperty.call(t,i[r]))return!1;if(sx&&e instanceof Element)return!1;for(r=n;r--!==0;)if(!((i[r]==="_owner"||i[r]==="__v"||i[r]==="__o")&&e.$$typeof)&&!na(e[i[r]],t[i[r]]))return!1;return!0}return e!==e&&t!==t}var cx=function(t,n){try{return na(t,n)}catch(r){if((r.message||"").match(/stack|recursion/i))return console.warn("react-fast-compare cannot handle circular refs"),!1;throw r}};const px=ar(cx);var hx=function(e,t,n,r,i,o,a,s){if(!e){var l;if(t===void 0)l=new Error("Minified exception occurred; use the non-minified dev environment for the full error message and additional helpful warnings.");else{var u=[n,r,i,o,a,s],c=0;l=new Error(t.replace(/%s/g,function(){return u[c++]})),l.name="Invariant Violation"}throw l.framesToPop=1,l}},mx=hx;const lp=ar(mx);var fx=function(t,n,r,i){var o=r?r.call(i,t,n):void 0;if(o!==void 0)return!!o;if(t===n)return!0;if(typeof t!="object"||!t||typeof n!="object"||!n)return!1;var a=Object.keys(t),s=Object.keys(n);if(a.length!==s.length)return!1;for(var l=Object.prototype.hasOwnProperty.bind(n),u=0;u<a.length;u++){var c=a[u];if(!l(c))return!1;var d=t[c],h=n[c];if(o=r?r.call(i,d,h,c):void 0,o===!1||o===void 0&&d!==h)return!1}return!0};const gx=ar(fx);var Jf=(e=>(e.BASE="base",e.BODY="body",e.HEAD="head",e.HTML="html",e.LINK="link",e.META="meta",e.NOSCRIPT="noscript",e.SCRIPT="script",e.STYLE="style",e.TITLE="title",e.FRAGMENT="Symbol(react.fragment)",e))(Jf||{}),Ws={link:{rel:["amphtml","canonical","alternate"]},script:{type:["application/ld+json"]},meta:{charset:"",name:["generator","robots","description"],property:["og:type","og:title","og:url","og:image","og:image:alt","og:description","twitter:url","twitter:title","twitter:description","twitter:image","twitter:image:alt","twitter:card","twitter:site"]}},up=Object.values(Jf),wd={accesskey:"accessKey",charset:"charSet",class:"className",contenteditable:"contentEditable",contextmenu:"contextMenu","http-equiv":"httpEquiv",itemprop:"itemProp",tabindex:"tabIndex"},_x=Object.entries(wd).reduce((e,[t,n])=>(e[n]=t,e),{}),Dt="data-rh",Ir={DEFAULT_TITLE:"defaultTitle",DEFER:"defer",ENCODE_SPECIAL_CHARACTERS:"encodeSpecialCharacters",ON_CHANGE_CLIENT_STATE:"onChangeClientState",TITLE_TEMPLATE:"titleTemplate",PRIORITIZE_SEO_TAGS:"prioritizeSeoTags"},Dr=(e,t)=>{for(let n=e.length-1;n>=0;n-=1){const r=e[n];if(Object.prototype.hasOwnProperty.call(r,t))return r[t]}return null},yx=e=>{let t=Dr(e,"title");const n=Dr(e,Ir.TITLE_TEMPLATE);if(Array.isArray(t)&&(t=t.join("")),n&&t)return n.replace(/%s/g,()=>t);const r=Dr(e,Ir.DEFAULT_TITLE);return t||r||void 0},xx=e=>Dr(e,Ir.ON_CHANGE_CLIENT_STATE)||(()=>{}),Vs=(e,t)=>t.filter(n=>typeof n[e]<"u").map(n=>n[e]).reduce((n,r)=>({...n,...r}),{}),bx=(e,t)=>t.filter(n=>typeof n.base<"u").map(n=>n.base).reverse().reduce((n,r)=>{if(!n.length){const i=Object.keys(r);for(let o=0;o<i.length;o+=1){const s=i[o].toLowerCase();if(e.indexOf(s)!==-1&&r[s])return n.concat(r)}}return n},[]),vx=e=>console&&typeof console.warn=="function"&&console.warn(e),ci=(e,t,n)=>{const r={};return n.filter(i=>Array.isArray(i[e])?!0:(typeof i[e]<"u"&&vx(`Helmet: ${e} should be of type "Array". Instead found type "${typeof i[e]}"`),!1)).map(i=>i[e]).reverse().reduce((i,o)=>{const a={};o.filter(l=>{let u;const c=Object.keys(l);for(let h=0;h<c.length;h+=1){const m=c[h],f=m.toLowerCase();t.indexOf(f)!==-1&&!(u==="rel"&&l[u].toLowerCase()==="canonical")&&!(f==="rel"&&l[f].toLowerCase()==="stylesheet")&&(u=f),t.indexOf(m)!==-1&&(m==="innerHTML"||m==="cssText"||m==="itemprop")&&(u=m)}if(!u||!l[u])return!1;const d=l[u].toLowerCase();return r[u]||(r[u]={}),a[u]||(a[u]={}),r[u][d]?!1:(a[u][d]=!0,!0)}).reverse().forEach(l=>i.push(l));const s=Object.keys(a);for(let l=0;l<s.length;l+=1){const u=s[l],c={...r[u],...a[u]};r[u]=c}return i},[]).reverse()},kx=(e,t)=>{if(Array.isArray(e)&&e.length){for(let n=0;n<e.length;n+=1)if(e[n][t])return!0}return!1},wx=e=>({baseTag:bx(["href"],e),bodyAttributes:Vs("bodyAttributes",e),defer:Dr(e,Ir.DEFER),encode:Dr(e,Ir.ENCODE_SPECIAL_CHARACTERS),htmlAttributes:Vs("htmlAttributes",e),linkTags:ci("link",["rel","href"],e),metaTags:ci("meta",["name","charset","http-equiv","property","itemprop"],e),noscriptTags:ci("noscript",["innerHTML"],e),onChangeClientState:xx(e),scriptTags:ci("script",["src","innerHTML"],e),styleTags:ci("style",["cssText"],e),title:yx(e),titleAttributes:Vs("titleAttributes",e),prioritizeSeoTags:kx(e,Ir.PRIORITIZE_SEO_TAGS)}),e0=e=>Array.isArray(e)?e.join(""):e,Tx=(e,t)=>{const n=Object.keys(e);for(let r=0;r<n.length;r+=1)if(t[n[r]]&&t[n[r]].includes(e[n[r]]))return!0;return!1},Bs=(e,t)=>Array.isArray(e)?e.reduce((n,r)=>(Tx(r,t)?n.priority.push(r):n.default.push(r),n),{priority:[],default:[]}):{default:e,priority:[]},dp=(e,t)=>({...e,[t]:void 0}),Cx=["noscript","script","style"],uu=(e,t=!0)=>t===!1?String(e):String(e).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#x27;"),t0=e=>Object.keys(e).reduce((t,n)=>{const r=typeof e[n]<"u"?`${n}="${e[n]}"`:`${n}`;return t?`${t} ${r}`:r},""),Sx=(e,t,n,r)=>{const i=t0(n),o=e0(t);return i?`<${e} ${Dt}="true" ${i}>${uu(o,r)}</${e}>`:`<${e} ${Dt}="true">${uu(o,r)}</${e}>`},zx=(e,t,n=!0)=>t.reduce((r,i)=>{const o=i,a=Object.keys(o).filter(u=>!(u==="innerHTML"||u==="cssText")).reduce((u,c)=>{const d=typeof o[c]>"u"?c:`${c}="${uu(o[c],n)}"`;return u?`${u} ${d}`:d},""),s=o.innerHTML||o.cssText||"",l=Cx.indexOf(e)===-1;return`${r}<${e} ${Dt}="true" ${a}${l?"/>":`>${s}</${e}>`}`},""),n0=(e,t={})=>Object.keys(e).reduce((n,r)=>{const i=wd[r];return n[i||r]=e[r],n},t),Ex=(e,t,n)=>{const r={key:t,[Dt]:!0},i=n0(n,r);return[Re.createElement("title",i,t)]},ra=(e,t)=>t.map((n,r)=>{const i={key:r,[Dt]:!0};return Object.keys(n).forEach(o=>{const s=wd[o]||o;if(s==="innerHTML"||s==="cssText"){const l=n.innerHTML||n.cssText;i.dangerouslySetInnerHTML={__html:l}}else i[s]=n[o]}),Re.createElement(e,i)}),bt=(e,t,n=!0)=>{switch(e){case"title":return{toComponent:()=>Ex(e,t.title,t.titleAttributes),toString:()=>Sx(e,t.title,t.titleAttributes,n)};case"bodyAttributes":case"htmlAttributes":return{toComponent:()=>n0(t),toString:()=>t0(t)};default:return{toComponent:()=>ra(e,t),toString:()=>zx(e,t,n)}}},Px=({metaTags:e,linkTags:t,scriptTags:n,encode:r})=>{const i=Bs(e,Ws.meta),o=Bs(t,Ws.link),a=Bs(n,Ws.script);return{priorityMethods:{toComponent:()=>[...ra("meta",i.priority),...ra("link",o.priority),...ra("script",a.priority)],toString:()=>`${bt("meta",i.priority,r)} ${bt("link",o.priority,r)} ${bt("script",a.priority,r)}`},metaTags:i.default,linkTags:o.default,scriptTags:a.default}},Nx=e=>{const{baseTag:t,bodyAttributes:n,encode:r=!0,htmlAttributes:i,noscriptTags:o,styleTags:a,title:s="",titleAttributes:l,prioritizeSeoTags:u}=e;let{linkTags:c,metaTags:d,scriptTags:h}=e,m={toComponent:()=>{},toString:()=>""};return u&&({priorityMethods:m,linkTags:c,metaTags:d,scriptTags:h}=Px(e)),{priority:m,base:bt("base",t,r),bodyAttributes:bt("bodyAttributes",n,r),htmlAttributes:bt("htmlAttributes",i,r),link:bt("link",c,r),meta:bt("meta",d,r),noscript:bt("noscript",o,r),script:bt("script",h,r),style:bt("style",a,r),title:bt("title",{title:s,titleAttributes:l},r)}},du=Nx,Fo=[],r0=!!(typeof window<"u"&&window.document&&window.document.createElement),cu=class{constructor(e,t){Ie(this,"instances",[]);Ie(this,"canUseDOM",r0);Ie(this,"context");Ie(this,"value",{setHelmet:e=>{this.context.helmet=e},helmetInstances:{get:()=>this.canUseDOM?Fo:this.instances,add:e=>{(this.canUseDOM?Fo:this.instances).push(e)},remove:e=>{const t=(this.canUseDOM?Fo:this.instances).indexOf(e);(this.canUseDOM?Fo:this.instances).splice(t,1)}}});this.context=e,this.canUseDOM=t||!1,t||(e.helmet=du({baseTag:[],bodyAttributes:{},htmlAttributes:{},linkTags:[],metaTags:[],noscriptTags:[],scriptTags:[],styleTags:[],title:"",titleAttributes:{}}))}},Ax={},i0=Re.createContext(Ax),Gn,o0=(Gn=class extends C.Component{constructor(n){super(n);Ie(this,"helmetData");this.helmetData=new cu(this.props.context||{},Gn.canUseDOM)}render(){return Re.createElement(i0.Provider,{value:this.helmetData.value},this.props.children)}},Ie(Gn,"canUseDOM",r0),Gn),mr=(e,t)=>{const n=document.head||document.querySelector("head"),r=n.querySelectorAll(`${e}[${Dt}]`),i=[].slice.call(r),o=[];let a;return t&&t.length&&t.forEach(s=>{const l=document.createElement(e);for(const u in s)if(Object.prototype.hasOwnProperty.call(s,u))if(u==="innerHTML")l.innerHTML=s.innerHTML;else if(u==="cssText")l.styleSheet?l.styleSheet.cssText=s.cssText:l.appendChild(document.createTextNode(s.cssText));else{const c=u,d=typeof s[c]>"u"?"":s[c];l.setAttribute(u,d)}l.setAttribute(Dt,"true"),i.some((u,c)=>(a=c,l.isEqualNode(u)))?i.splice(a,1):o.push(l)}),i.forEach(s=>{var l;return(l=s.parentNode)==null?void 0:l.removeChild(s)}),o.forEach(s=>n.appendChild(s)),{oldTags:i,newTags:o}},pu=(e,t)=>{const n=document.getElementsByTagName(e)[0];if(!n)return;const r=n.getAttribute(Dt),i=r?r.split(","):[],o=[...i],a=Object.keys(t);for(const s of a){const l=t[s]||"";n.getAttribute(s)!==l&&n.setAttribute(s,l),i.indexOf(s)===-1&&i.push(s);const u=o.indexOf(s);u!==-1&&o.splice(u,1)}for(let s=o.length-1;s>=0;s-=1)n.removeAttribute(o[s]);i.length===o.length?n.removeAttribute(Dt):n.getAttribute(Dt)!==a.join(",")&&n.setAttribute(Dt,a.join(","))},Lx=(e,t)=>{typeof e<"u"&&document.title!==e&&(document.title=e0(e)),pu("title",t)},cp=(e,t)=>{const{baseTag:n,bodyAttributes:r,htmlAttributes:i,linkTags:o,metaTags:a,noscriptTags:s,onChangeClientState:l,scriptTags:u,styleTags:c,title:d,titleAttributes:h}=e;pu("body",r),pu("html",i),Lx(d,h);const m={baseTag:mr("base",n),linkTags:mr("link",o),metaTags:mr("meta",a),noscriptTags:mr("noscript",s),scriptTags:mr("script",u),styleTags:mr("style",c)},f={},x={};Object.keys(m).forEach(S=>{const{newTags:g,oldTags:y}=m[S];g.length&&(f[S]=g),y.length&&(x[S]=m[S].oldTags)}),t&&t(),l(e,f,x)},pi=null,jx=e=>{pi&&cancelAnimationFrame(pi),e.defer?pi=requestAnimationFrame(()=>{cp(e,()=>{pi=null})}):(cp(e),pi=null)},Rx=jx,pp=class extends C.Component{constructor(){super(...arguments);Ie(this,"rendered",!1)}shouldComponentUpdate(t){return!gx(t,this.props)}componentDidUpdate(){this.emitChange()}componentWillUnmount(){const{helmetInstances:t}=this.props.context;t.remove(this),this.emitChange()}emitChange(){const{helmetInstances:t,setHelmet:n}=this.props.context;let r=null;const i=wx(t.get().map(o=>{const a={...o.props};return delete a.context,a}));o0.canUseDOM?Rx(i):du&&(r=du(i)),n(r)}init(){if(this.rendered)return;this.rendered=!0;const{helmetInstances:t}=this.props.context;t.add(this),this.emitChange()}render(){return this.init(),null}},cl,Fx=(cl=class extends C.Component{shouldComponentUpdate(e){return!px(dp(this.props,"helmetData"),dp(e,"helmetData"))}mapNestedChildrenToProps(e,t){if(!t)return null;switch(e.type){case"script":case"noscript":return{innerHTML:t};case"style":return{cssText:t};default:throw new Error(`<${e.type} /> elements are self-closing and can not contain children. Refer to our API for more information.`)}}flattenArrayTypeChildren(e,t,n,r){return{...t,[e.type]:[...t[e.type]||[],{...n,...this.mapNestedChildrenToProps(e,r)}]}}mapObjectTypeChildren(e,t,n,r){switch(e.type){case"title":return{...t,[e.type]:r,titleAttributes:{...n}};case"body":return{...t,bodyAttributes:{...n}};case"html":return{...t,htmlAttributes:{...n}};default:return{...t,[e.type]:{...n}}}}mapArrayTypeChildrenToProps(e,t){let n={...t};return Object.keys(e).forEach(r=>{n={...n,[r]:e[r]}}),n}warnOnInvalidChildren(e,t){return lp(up.some(n=>e.type===n),typeof e.type=="function"?"You may be attempting to nest <Helmet> components within each other, which is not allowed. Refer to our API for more information.":`Only elements types ${up.join(", ")} are allowed. Helmet does not support rendering <${e.type}> elements. Refer to our API for more information.`),lp(!t||typeof t=="string"||Array.isArray(t)&&!t.some(n=>typeof n!="string"),`Helmet expects a string as a child of <${e.type}>. Did you forget to wrap your children in braces? ( <${e.type}>{\`\`}</${e.type}> ) Refer to our API for more information.`),!0}mapChildrenToProps(e,t){let n={};return Re.Children.forEach(e,r=>{if(!r||!r.props)return;const{children:i,...o}=r.props,a=Object.keys(o).reduce((l,u)=>(l[_x[u]||u]=o[u],l),{});let{type:s}=r;switch(typeof s=="symbol"?s=s.toString():this.warnOnInvalidChildren(r,i),s){case"Symbol(react.fragment)":t=this.mapChildrenToProps(i,t);break;case"link":case"meta":case"noscript":case"script":case"style":n=this.flattenArrayTypeChildren(r,n,a,i);break;default:t=this.mapObjectTypeChildren(r,t,a,i);break}}),this.mapArrayTypeChildrenToProps(n,t)}render(){const{children:e,...t}=this.props;let n={...t},{helmetData:r}=t;if(e&&(n=this.mapChildrenToProps(e,n)),r&&!(r instanceof cu)){const i=r;r=new cu(i.context,!0),delete n.helmetData}return r?Re.createElement(pp,{...n,context:r.value}):Re.createElement(i0.Consumer,null,i=>Re.createElement(pp,{...n,context:i}))}},Ie(cl,"defaultProps",{defer:!0,encodeSpecialCharacters:!0,prioritizeSeoTags:!1}),cl);const Ix={"numpy-array-creation":{note:"Use the torch creation functions: torch.zeros, torch.ones, torch.arange, torch.linspace, torch.eye.",starterCode:`import torch

def create_arrays() -> dict:
    """
    Create various tensors.

    Returns:
        Dictionary with keys: zeros, ones, arange, linspace, eye
    """
    # Your code here
    pass
`,solution:`import torch

def create_arrays() -> dict:
    return {
        'zeros': torch.zeros(3, 4),
        'ones': torch.ones(2, 3),
        'arange': torch.arange(10),
        'linspace': torch.linspace(0, 1, 5),
        'eye': torch.eye(4),
    }
`,testCases:[{id:"1",description:"Zeros shape correct",input:'tuple(create_arrays()["zeros"].shape)',expected:"(3, 4)",hidden:!1},{id:"2",description:"Linspace values",input:'create_arrays()["linspace"].tolist()',expected:"[0.0, 0.25, 0.5, 0.75, 1.0]",hidden:!1},{id:"3",description:"Arange values",input:'create_arrays()["arange"].tolist()',expected:"[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]",hidden:!1},{id:"4",description:"Eye is identity",input:'bool(torch.equal(create_arrays()["eye"], torch.eye(4)))',expected:"True",hidden:!0},{id:"5",description:"Ones and zeros have the right values",input:'create_arrays()["ones"].tolist() + create_arrays()["zeros"][0].tolist()',expected:"[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.0, 0.0, 0.0, 0.0]",hidden:!0}],hints:["torch.zeros(3, 4) takes the sizes as separate arguments (a tuple works too)","torch.linspace(start, end, steps) includes both endpoints","torch.eye(n) creates an identity matrix"]},"numpy-indexing":{note:"Tensors support the same slicing as NumPy, except negative steps: use torch.flip instead of arr[::-1].",starterCode:`import torch

def advanced_indexing(arr: torch.Tensor) -> dict:
    """
    Perform advanced indexing operations.

    Args:
        arr: 2D input tensor

    Returns:
        Dictionary with indexed results
    """
    # Your code here
    pass
`,solution:`import torch

def advanced_indexing(arr: torch.Tensor) -> dict:
    return {
        'row_1': arr[1],
        'col_2': arr[:, 2],
        'subarray': arr[1:3, 2:5],
        'diagonal': torch.diag(arr),
        'reversed': torch.flip(arr, dims=[0]),
    }
`,testCases:[{id:"1",description:"Row extraction",input:'advanced_indexing(torch.arange(20).reshape(4, 5))["row_1"].tolist()',expected:"[5, 6, 7, 8, 9]",hidden:!1},{id:"2",description:"Subarray correct",input:'advanced_indexing(torch.arange(20).reshape(4, 5))["subarray"].tolist()',expected:"[[7, 8, 9], [12, 13, 14]]",hidden:!1},{id:"3",description:"Column extraction",input:'advanced_indexing(torch.arange(20).reshape(4, 5))["col_2"].tolist()',expected:"[2, 7, 12, 17]",hidden:!0},{id:"4",description:"Diagonal",input:'advanced_indexing(torch.arange(20).reshape(4, 5))["diagonal"].tolist()',expected:"[0, 6, 12, 18]",hidden:!0},{id:"5",description:"Rows reversed",input:'advanced_indexing(torch.arange(20).reshape(4, 5))["reversed"][:, 0].tolist()',expected:"[15, 10, 5, 0]",hidden:!0}],hints:["arr[1] gets row 1, arr[:, 2] gets column 2","arr[1:3, 2:5] gets a subarray","torch.diag(arr) extracts the main diagonal","torch.flip(arr, dims=[0]) reverses the rows (arr[::-1] raises an error for tensors)"]},"numpy-broadcasting":{note:"Reductions take dim= and keepdim= (instead of axis= and keepdims=).",starterCode:`import torch

def broadcasting_ops(arr: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> dict:
    """
    Perform broadcasting operations.

    Args:
        arr: 2D float tensor of shape (3, 4)
        bias: 1D tensor of shape (4,)
        scale: 2D tensor of shape (3, 1)

    Returns:
        Dictionary with results of broadcasting operations
    """
    # Your code here
    pass
`,solution:`import torch

def broadcasting_ops(arr: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> dict:
    # Row centered: subtract mean of each row
    row_means = arr.mean(dim=1, keepdim=True)  # (3, 1)
    row_centered = arr - row_means

    # Column centered: subtract mean of each column
    col_means = arr.mean(dim=0)  # (4,)
    col_centered = arr - col_means

    # Add bias to each row: (4,) broadcasts to (3, 4)
    biased = arr + bias

    # Scale each row: (3, 1) broadcasts to (3, 4)
    scaled = arr * scale

    return {
        'row_centered': row_centered,
        'col_centered': col_centered,
        'biased': biased,
        'scaled': scaled,
    }
`,testCases:[{id:"1",description:"Row centering",input:'broadcasting_ops(torch.arange(1, 13).float().reshape(3, 4), torch.zeros(4), torch.ones(3, 1))["row_centered"].tolist()',expected:"[[-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 0.5, 1.5]]",hidden:!1},{id:"2",description:"Bias addition",input:'broadcasting_ops(torch.arange(1, 13).float().reshape(3, 4), torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.ones(3, 1))["biased"].tolist()',expected:"[[2.0, 4.0, 6.0, 8.0], [6.0, 8.0, 10.0, 12.0], [10.0, 12.0, 14.0, 16.0]]",hidden:!1},{id:"3",description:"Column centering",input:'broadcasting_ops(torch.arange(1, 13).float().reshape(3, 4), torch.zeros(4), torch.ones(3, 1))["col_centered"].tolist()',expected:"[[-4.0, -4.0, -4.0, -4.0], [0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0]]",hidden:!0},{id:"4",description:"Row scaling",input:'broadcasting_ops(torch.arange(1, 13).float().reshape(3, 4), torch.zeros(4), torch.tensor([[1.0], [2.0], [3.0]]))["scaled"].tolist()',expected:"[[1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0], [27.0, 30.0, 33.0, 36.0]]",hidden:!0}],hints:["Row means: arr.mean(dim=1, keepdim=True) has shape (3, 1)","Column means: arr.mean(dim=0) has shape (4,)","Bias shape (4,) broadcasts with arr shape (3, 4)","Scale shape (3, 1) broadcasts with arr shape (3, 4)"]},"numpy-aggregations":{note:"torch.std is unbiased by default; pass unbiased=False to match NumPy. Tensors have no unravel_index helper: convert the flat argmax yourself.",starterCode:`import torch

def compute_aggregations(arr: torch.Tensor) -> dict:
    """
    Compute aggregation statistics.

    Args:
        arr: 2D float tensor

    Returns:
        Dictionary with aggregation results
    """
    # Your code here
    pass
`,solution:`import torch

def compute_aggregations(arr: torch.Tensor) -> dict:
    n_cols = arr.shape[1]
    flat_max = arr.argmax().item()
    flat_min = arr.argmin().item()
    return {
        'global_mean': arr.mean(),
        'global_std': arr.std(unbiased=False),  # population std, like np.std
        'global_min': arr.min(),
        'global_max': arr.max(),
        'global_sum': arr.sum(),
        'row_mean': arr.mean(dim=1),
        'row_sum': arr.sum(dim=1),
        'col_mean': arr.mean(dim=0),
        'col_sum': arr.sum(dim=0),
        'argmax': (flat_max // n_cols, flat_max % n_cols),
        'argmin': (flat_min // n_cols, flat_min % n_cols),
    }
`,testCases:[{id:"1",description:"Global mean",input:'compute_aggregations(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))["global_mean"]',expected:"3.5",hidden:!1},{id:"2",description:"Row sums",input:'compute_aggregations(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))["row_sum"].tolist()',expected:"[6.0, 15.0]",hidden:!1},{id:"3",description:"Column means",input:'compute_aggregations(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))["col_mean"].tolist()',expected:"[2.5, 3.5, 4.5]",hidden:!0},{id:"4",description:"Population standard deviation",input:'round(float(compute_aggregations(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))["global_std"]), 4)',expected:"1.7078",hidden:!0},{id:"5",description:"2D argmax and argmin",input:'(lambda r: (tuple(int(i) for i in r["argmax"]), tuple(int(i) for i in r["argmin"])))(compute_aggregations(torch.tensor([[3.0, 9.0, 1.0], [4.0, 0.0, 6.0]])))',expected:"((0, 1), (1, 1))",hidden:!0}],hints:["arr.mean() computes the global mean; arr.mean(dim=0) the column means","arr.std(unbiased=False) matches np.std","arr.argmax() gives the index in the flattened tensor: row = idx // n_cols, col = idx % n_cols"]},"numpy-reshape-transpose":{note:"torch uses .view/.reshape, .T, .permute(...) (the equivalent of np.transpose with axes) and .flatten().",starterCode:`import torch

def reshape_transpose(arr: torch.Tensor) -> dict:
    """
    Perform reshape and transpose operations.

    Args:
        arr: 1D tensor of 24 elements

    Returns:
        Dictionary with reshaped tensors
    """
    # Your code here
    pass
`,solution:`import torch

def reshape_transpose(arr: torch.Tensor) -> dict:
    # 2D reshape
    arr_2d = arr.view(4, 6)

    # 3D reshape
    arr_3d = arr.view(2, 3, 4)

    # Transpose 2D
    arr_2d_T = arr_2d.T

    # Reverse the axes of the 3D tensor
    arr_3d_swapped = arr_3d.permute(2, 1, 0)

    # Flatten
    arr_flat = arr_2d.flatten()

    return {
        'arr_2d': arr_2d,
        'arr_3d': arr_3d,
        'arr_2d_transposed': arr_2d_T,
        'arr_3d_swapped': arr_3d_swapped,
        'arr_flat': arr_flat,
    }
`,testCases:[{id:"1",description:"2D reshape shape",input:'tuple(reshape_transpose(torch.arange(24))["arr_2d"].shape)',expected:"(4, 6)",hidden:!1},{id:"2",description:"3D reshape shape",input:'tuple(reshape_transpose(torch.arange(24))["arr_3d"].shape)',expected:"(2, 3, 4)",hidden:!1},{id:"3",description:"Transposed values",input:'reshape_transpose(torch.arange(24))["arr_2d_transposed"][:2].tolist()',expected:"[[0, 6, 12, 18], [1, 7, 13, 19]]",hidden:!0},{id:"4",description:"Axes reversed on the 3D tensor",input:'reshape_transpose(torch.arange(24))["arr_3d_swapped"][:, 0, :].tolist()',expected:"[[0, 12], [1, 13], [2, 14], [3, 15]]",hidden:!0},{id:"5",description:"Flattened back to the original order",input:'reshape_transpose(torch.arange(24))["arr_flat"].tolist() == list(range(24))',expected:"True",hidden:!0}],hints:["arr.view(4, 6) or arr.reshape(4, -1)","arr.view(2, 3, 4) for 3D",".T transposes a 2D tensor; .permute(2, 1, 0) reorders the axes of a 3D tensor",".flatten() gives back a 1D tensor"]},"numpy-array-sum":{note:"Use torch.sum or the .sum() method. A 0-d tensor result is fine (tests call .item() for you).",starterCode:`import torch

def array_sum(arr: torch.Tensor) -> float:
    """
    Calculate the sum of all elements in a tensor.

    Args:
        arr: A tensor of numbers

    Returns:
        The sum of all elements
    """
    # Your code here
    pass
`,solution:`import torch

def array_sum(arr: torch.Tensor) -> float:
    """
    Calculate the sum of all elements in a tensor.
    """
    return torch.sum(arr)
`,testCases:[{id:"1",description:"Basic positive numbers",input:"array_sum(torch.tensor([1, 2, 3, 4, 5]))",expected:"15",hidden:!1},{id:"2",description:"With negatives",input:"array_sum(torch.tensor([-1, 0, 1]))",expected:"0",hidden:!1},{id:"3",description:"Single element",input:"array_sum(torch.tensor([42]))",expected:"42",hidden:!1},{id:"4",description:"Larger array",input:"array_sum(torch.arange(1, 11))",expected:"55",hidden:!0},{id:"5",description:"Float values",input:"array_sum(torch.tensor([0.5, 1.5, -1.0]))",expected:"1.0",hidden:!0}],hints:["Tensors have built-in reductions, just like NumPy arrays.","Try torch.sum(arr) or arr.sum()."]},"numpy-matrix-multiply":{note:"Use torch.matmul, torch.mm or the @ operator, not Python loops.",starterCode:`import torch

def matrix_multiply(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Multiply two matrices A and B.

    Args:
        A: First matrix of shape (m, n)
        B: Second matrix of shape (n, p)

    Returns:
        Result matrix of shape (m, p)
    """
    # Your code here
    pass
`,solution:`import torch

def matrix_multiply(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Multiply two matrices A and B.
    """
    return A @ B
    # Alternatives: torch.matmul(A, B), torch.mm(A, B)
`,testCases:[{id:"1",description:"2x2 matrices",input:"matrix_multiply(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]))",expected:"[[19, 22], [43, 50]]",hidden:!1},{id:"2",description:"Identity multiplication",input:"matrix_multiply(torch.tensor([[1, 0], [0, 1]]), torch.tensor([[5, 6], [7, 8]]))",expected:"[[5, 6], [7, 8]]",hidden:!1},{id:"3",description:"Different dimensions",input:"matrix_multiply(torch.tensor([[1, 2, 3]]), torch.tensor([[1], [2], [3]]))",expected:"[[14]]",hidden:!0},{id:"4",description:"Float matrices, non-square",input:"matrix_multiply(torch.tensor([[1.0, 2.0], [0.5, -1.0], [0.0, 3.0]]), torch.tensor([[2.0, 0.0, 1.0], [1.0, 1.0, 0.0]]))",expected:"[[4.0, 2.0, 1.0], [0.0, -1.0, 0.5], [3.0, 3.0, 0.0]]",hidden:!0}],hints:["PyTorch provides several ways to do matrix multiplication.","Try torch.matmul(), torch.mm(), or the @ operator."]},"numpy-broadcast-add":{note:"Tensors broadcast with the same rules as NumPy arrays.",starterCode:`import torch

def broadcast_add(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """
    Add a vector to each row of a matrix using broadcasting.

    Args:
        matrix: 2D tensor of shape (m, n)
        vector: 1D tensor of shape (n,)

    Returns:
        Result matrix of shape (m, n)
    """
    # Your code here
    pass
`,solution:`import torch

def broadcast_add(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """
    Add a vector to each row of a matrix using broadcasting.
    """
    return matrix + vector
`,testCases:[{id:"1",description:"Basic broadcasting",input:"broadcast_add(torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([10, 20, 30]))",expected:"[[11, 22, 33], [14, 25, 36]]",hidden:!1},{id:"2",description:"With zeros",input:"broadcast_add(torch.tensor([[1, 2], [3, 4]]), torch.tensor([0, 0]))",expected:"[[1, 2], [3, 4]]",hidden:!1},{id:"3",description:"Negative values",input:"broadcast_add(torch.tensor([[1, 2], [3, 4]]), torch.tensor([-1, -2]))",expected:"[[0, 0], [2, 2]]",hidden:!0},{id:"4",description:"Float values, 3 rows",input:"broadcast_add(torch.ones(3, 2), torch.tensor([0.5, -1.0]))",expected:"[[1.5, 0.0], [1.5, 0.0], [1.5, 0.0]]",hidden:!0}],hints:["PyTorch automatically broadcasts the vector across rows.","Simply use the + operator."]}},Dx={"einsum-basics":{note:"torch.einsum takes the same subscript strings as np.einsum.",starterCode:`import torch

def einsum_basics(arr: torch.Tensor) -> dict:
    """
    Perform basic einsum operations.

    Args:
        arr: 2D input tensor

    Returns:
        Dictionary with einsum results
    """
    # Your code here
    pass
`,solution:`import torch

def einsum_basics(arr: torch.Tensor) -> dict:
    return {
        'sum_all': torch.einsum('ij->', arr),
        'row_sum': torch.einsum('ij->i', arr),
        'col_sum': torch.einsum('ij->j', arr),
        'transpose': torch.einsum('ij->ji', arr),
        'diagonal': torch.einsum('ii->i', arr) if arr.shape[0] == arr.shape[1] else None,
    }
`,testCases:[{id:"1",description:"Sum all elements",input:'einsum_basics(torch.tensor([[1, 2], [3, 4]]))["sum_all"]',expected:"10",hidden:!1},{id:"2",description:"Transpose",input:'einsum_basics(torch.tensor([[1, 2], [3, 4]]))["transpose"]',expected:"[[1, 3], [2, 4]]",hidden:!1},{id:"3",description:"Row and column sums",input:'(lambda r: (r["row_sum"].tolist(), r["col_sum"].tolist()))(einsum_basics(torch.tensor([[1, 2, 3], [4, 5, 6]])))',expected:"([6, 15], [5, 7, 9])",hidden:!0},{id:"4",description:"Diagonal of a square matrix",input:'einsum_basics(torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))["diagonal"]',expected:"[1, 5, 9]",hidden:!0},{id:"5",description:"No diagonal for a non-square matrix",input:'einsum_basics(torch.ones(2, 3))["diagonal"] is None',expected:"True",hidden:!0}],hints:["'ij->' sums all elements","'ij->i' sums each row","'ij->ji' transposes","'ii->i' extracts the diagonal (square matrices only)"]},"einsum-matrix-ops":{note:"Use torch.einsum for every entry.",starterCode:`import torch

def einsum_matrix_ops(A: torch.Tensor, B: torch.Tensor, v: torch.Tensor) -> dict:
    """
    Perform matrix operations using einsum.

    Args:
        A: Matrix of shape (m, k)
        B: Matrix of shape (k, n)
        v: Vector of shape (k,)

    Returns:
        Dictionary with operation results
    """
    # Your code here
    pass
`,solution:`import torch

def einsum_matrix_ops(A: torch.Tensor, B: torch.Tensor, v: torch.Tensor) -> dict:
    return {
        'matmul': torch.einsum('ik,kj->ij', A, B),
        'matvec': torch.einsum('ij,j->i', A, v),
        'outer_product': torch.einsum('i,j->ij', v, v),
        'hadamard': torch.einsum('ij,ij->ij', A, A),  # A * A
        'frobenius': torch.einsum('ij,ij->', A, A),   # (A * A).sum()
        'trace': torch.einsum('ii->', A) if A.shape[0] == A.shape[1] else None,
    }
`,testCases:[{id:"1",description:"Matrix multiplication",input:'einsum_matrix_ops(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]), torch.tensor([1, 2]))["matmul"]',expected:"[[19, 22], [43, 50]]",hidden:!1},{id:"2",description:"Outer product",input:'einsum_matrix_ops(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]), torch.tensor([1, 2]))["outer_product"]',expected:"[[1, 2], [2, 4]]",hidden:!1},{id:"3",description:"Matrix-vector product and Hadamard",input:'(lambda r: (r["matvec"].tolist(), r["hadamard"].tolist()))(einsum_matrix_ops(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]), torch.tensor([1, 2])))',expected:"([5, 11], [[1, 4], [9, 16]])",hidden:!0},{id:"4",description:"Frobenius inner product and trace",input:'(lambda r: (int(r["frobenius"]), int(r["trace"])))(einsum_matrix_ops(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]), torch.tensor([1, 2])))',expected:"(30, 5)",hidden:!0},{id:"5",description:"Non-square A: no trace",input:'(lambda r: (r["matmul"].tolist(), r["trace"] is None))(einsum_matrix_ops(torch.tensor([[1, 0, 2]]), torch.tensor([[1], [2], [3]]), torch.tensor([1, 1, 1])))',expected:"([[7]], True)",hidden:!0}],hints:["'ik,kj->ij' contracts over k (matrix multiply)","'ij,j->i' contracts the vector with the columns","'i,j->ij' creates an outer product (no contraction)"]},"einsum-batch-ops":{note:"Use torch.einsum for the two batched products; F.softmax(x, dim=-1) is fine for the softmax.",starterCode:`import torch
import torch.nn.functional as F

def einsum_batch_ops(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> dict:
    """
    Perform batch operations using einsum.

    Args:
        Q: Queries of shape (batch, seq_q, dim)
        K: Keys of shape (batch, seq_k, dim)
        V: Values of shape (batch, seq_k, dim_v)

    Returns:
        Dictionary with batch operation results
    """
    # Your code here
    pass
`,solution:`import math
import torch
import torch.nn.functional as F

def einsum_batch_ops(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> dict:
    # Attention scores: Q @ K.T for each batch -> (batch, seq_q, seq_k)
    scores = torch.einsum('bqd,bkd->bqk', Q, K)

    # Scale and normalize over the keys
    d_k = Q.shape[-1]
    attention_weights = F.softmax(scores / math.sqrt(d_k), dim=-1)

    # Weighted sum of values -> (batch, seq_q, dim_v)
    output = torch.einsum('bqk,bkv->bqv', attention_weights, V)

    return {
        'scores': scores,
        'attention_weights': attention_weights,
        'output': output,
    }
`,testCases:[{id:"1",description:"Attention scores shape",input:'tuple(einsum_batch_ops(torch.ones(2, 4, 8), torch.ones(2, 6, 8), torch.ones(2, 6, 16))["scores"].shape)',expected:"(2, 4, 6)",hidden:!1},{id:"2",description:"Output shape",input:'tuple(einsum_batch_ops(torch.ones(2, 4, 8), torch.ones(2, 6, 8), torch.ones(2, 6, 16))["output"].shape)',expected:"(2, 4, 16)",hidden:!1},{id:"3",description:"Scores equal Q @ K^T",input:`(lambda: (
    Q := torch.randn(2, 3, 4),
    K := torch.randn(2, 5, 4),
    V := torch.randn(2, 5, 6),
    bool(torch.allclose(einsum_batch_ops(Q, K, V)["scores"], Q @ K.transpose(1, 2), atol=1e-5))
)[-1])()`,expected:"True",hidden:!0},{id:"4",description:"Weights and output match scaled softmax attention",input:`(lambda: (
    Q := torch.randn(2, 3, 4),
    K := torch.randn(2, 5, 4),
    V := torch.randn(2, 5, 6),
    s := Q @ K.transpose(1, 2) / 2.0,
    w := s.exp() / s.exp().sum(dim=-1, keepdim=True),
    r := einsum_batch_ops(Q, K, V),
    bool(torch.allclose(r["attention_weights"], w, atol=1e-5) and torch.allclose(r["output"], w @ V, atol=1e-5))
)[-1])()`,expected:"True",hidden:!0}],hints:["'bqd,bkd->bqk' computes Q @ K.T for each batch","'bqk,bkv->bqv' applies the attention weights to the values","b is the batch dimension, preserved in the output"]},"einsum-advanced":{note:"Use .view/.reshape and torch.einsum; F.softmax(x, dim=-1) is fine. Don't use nn.MultiheadAttention or F.scaled_dot_product_attention.",starterCode:`import math
import torch
import torch.nn.functional as F

def multi_head_attention_einsum(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                 num_heads: int) -> torch.Tensor:
    """
    Multi-head attention using einsum.

    Args:
        Q: Queries (batch, seq_q, d_model)
        K: Keys (batch, seq_k, d_model)
        V: Values (batch, seq_k, d_model)
        num_heads: Number of attention heads

    Returns:
        output: (batch, seq_q, d_model)
    """
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # Your code here
    pass
`,solution:`import math
import torch
import torch.nn.functional as F

def multi_head_attention_einsum(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                 num_heads: int) -> torch.Tensor:
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # Split heads: (batch, seq, num_heads, d_k)
    Q = Q.reshape(batch, seq_q, num_heads, d_k)
    K = K.reshape(batch, seq_k, num_heads, d_k)
    V = V.reshape(batch, seq_k, num_heads, d_k)

    # Scores for all heads: (batch, heads, seq_q, seq_k)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / math.sqrt(d_k)
    attention = F.softmax(scores, dim=-1)

    # Weighted sum of values: (batch, seq_q, heads, d_k)
    output = torch.einsum('bhqk,bkhd->bqhd', attention, V)

    # Merge heads back: (batch, seq_q, d_model)
    return output.reshape(batch, seq_q, d_model)
`,testCases:[{id:"1",description:"Output shape preserved",input:"tuple(multi_head_attention_einsum(torch.ones(2, 4, 64), torch.ones(2, 4, 64), torch.ones(2, 4, 64), 8).shape)",expected:"(2, 4, 64)",hidden:!1},{id:"2",description:"One head equals single-head attention",input:`(lambda: (
    Q := torch.randn(2, 3, 8),
    K := torch.randn(2, 5, 8),
    V := torch.randn(2, 5, 8),
    s := Q @ K.transpose(1, 2) / 8 ** 0.5,
    ref := (s.exp() / s.exp().sum(dim=-1, keepdim=True)) @ V,
    bool(torch.allclose(multi_head_attention_einsum(Q, K, V, 1), ref, atol=1e-5))
)[-1])()`,expected:"True",hidden:!1},{id:"3",description:"Output is finite",input:"bool(not torch.isnan(multi_head_attention_einsum(torch.ones(2, 4, 64), torch.ones(2, 4, 64), torch.ones(2, 4, 64), 8)).any())",expected:"True",hidden:!0},{id:"4",description:"Each head attends independently (2 heads)",input:`(lambda: (
    Q := torch.randn(2, 3, 8),
    K := torch.randn(2, 5, 8),
    V := torch.randn(2, 5, 8),
    out := multi_head_attention_einsum(Q, K, V, 2),
    s := Q[:, :, 4:] @ K[:, :, 4:].transpose(1, 2) / 2.0,
    ref := (s.exp() / s.exp().sum(dim=-1, keepdim=True)) @ V[:, :, 4:],
    bool(torch.allclose(out[:, :, 4:], ref, atol=1e-5))
)[-1])()`,expected:"True",hidden:!0}],hints:["Reshape Q, K, V to (batch, seq, num_heads, d_k)","Use 'bqhd,bkhd->bhqk' for the attention scores per head","Use 'bhqk,bkhd->bqhd' for the weighted values","Reshape back to (batch, seq, d_model)"]},"einsum-vs-matmul":{note:"Keep the key name 'numpy' from the description; its values are the equivalent plain torch operations (torch.sum, A @ B, A.T, ...).",starterCode:`import torch

def einsum_equivalence(A: torch.Tensor, B: torch.Tensor) -> dict:
    """
    Compare einsum with traditional tensor operations.

    Args:
        A: First matrix (m, k)
        B: Second matrix (k, n)

    Returns:
        Dictionary with 'einsum' and 'numpy' (plain torch op) results for comparison
    """
    # Your code here
    pass
`,solution:`import torch

def einsum_equivalence(A: torch.Tensor, B: torch.Tensor) -> dict:
    results = {
        'sum': {
            'einsum': torch.einsum('ij->', A),
            'numpy': torch.sum(A),
        },
        'row_sum': {
            'einsum': torch.einsum('ij->i', A),
            'numpy': torch.sum(A, dim=1),
        },
        'matmul': {
            'einsum': torch.einsum('ik,kj->ij', A, B),
            'numpy': A @ B,
        },
        'transpose': {
            'einsum': torch.einsum('ij->ji', A),
            'numpy': A.T,
        },
        'hadamard_sum': {
            'einsum': torch.einsum('ij,ij->', A, A),
            'numpy': torch.sum(A * A),
        },
    }

    # Verify all match
    all_match = all(
        torch.allclose(v['einsum'], v['numpy'])
        for v in results.values()
    )

    return {'results': results, 'all_match': all_match}
`,testCases:[{id:"1",description:"All results match",input:'bool(einsum_equivalence(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))["all_match"])',expected:"True",hidden:!1},{id:"2",description:"Matrix multiplication matches",input:'einsum_equivalence(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))["results"]["matmul"]["einsum"]',expected:"[[19.0, 22.0], [43.0, 50.0]]",hidden:!1},{id:"3",description:"All five operations present",input:'sorted(einsum_equivalence(torch.ones(2, 3), torch.ones(3, 2))["results"].keys())',expected:'["hadamard_sum", "matmul", "row_sum", "sum", "transpose"]',hidden:!0},{id:"4",description:"Hadamard sum and row sums",input:'(lambda r: (float(r["hadamard_sum"]["einsum"]), r["row_sum"]["numpy"].tolist()))(einsum_equivalence(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))["results"])',expected:"(91.0, [6.0, 15.0])",hidden:!0}],hints:["Use torch.allclose() to compare floating point tensors","Remember 'ik,kj->ij' is matrix multiplication"]}},Mx={"tensor-creation":{note:"Use the real torch creation functions. torch.manual_seed(42) makes torch.randn reproducible.",starterCode:`import torch

def create_tensors() -> dict:
    """
    Create tensors using various methods.

    Returns:
        Dictionary with named tensors
    """
    torch.manual_seed(42)

    # Your code here
    pass
`,solution:`import torch

def create_tensors() -> dict:
    torch.manual_seed(42)

    return {
        'from_list': torch.tensor([1, 2, 3, 4], dtype=torch.float32),
        'zeros': torch.zeros(3, 4),
        'ones': torch.ones(2, 3),
        'randn': torch.randn(2, 3),
        'arange': torch.arange(10),
        'linspace': torch.linspace(0, 1, 5),
        'eye': torch.eye(4),
        'full': torch.full((2, 3), 7.0),
    }
`,testCases:[{id:"1",description:"Zeros shape",input:'tuple(create_tensors()["zeros"].shape)',expected:"(3, 4)",hidden:!1},{id:"2",description:"Random tensor shape",input:'tuple(create_tensors()["randn"].shape)',expected:"(2, 3)",hidden:!1},{id:"3",description:"from_list is float32 with the right values",input:'(lambda t: (t.dtype == torch.float32, t.tolist()))(create_tensors()["from_list"])',expected:"(True, [1.0, 2.0, 3.0, 4.0])",hidden:!0},{id:"4",description:"full, linspace and arange values",input:'(lambda r: (r["full"].tolist(), r["linspace"].tolist(), r["arange"].tolist()))(create_tensors())',expected:"([[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]], [0.0, 0.25, 0.5, 0.75, 1.0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])",hidden:!0},{id:"5",description:"Seeded randn is reproducible; eye is identity",input:'bool(torch.equal(create_tensors()["randn"], create_tensors()["randn"]) and torch.equal(create_tensors()["eye"], torch.eye(4)))',expected:"True",hidden:!0}],hints:["torch.tensor(data, dtype=torch.float32) builds a tensor from a list","torch.zeros(3, 4), torch.ones(2, 3), torch.randn(2, 3) take sizes as arguments","torch.full((2, 3), 7.0) fills a tensor with one value"]},"tensor-operations":{note:"Use tensor methods with dim= (not axis=). torch.std is unbiased by default; pass unbiased=False to match np.std.",starterCode:`import torch

def tensor_ops(x: torch.Tensor, y: torch.Tensor) -> dict:
    """
    Perform common tensor operations.

    Args:
        x: First float tensor (2, 3)
        y: Second float tensor (2, 3)

    Returns:
        Dictionary with operation results
    """
    # Your code here
    pass
`,solution:`import torch

def tensor_ops(x: torch.Tensor, y: torch.Tensor) -> dict:
    return {
        # Arithmetic
        'add': x + y,
        'sub': x - y,
        'mul': x * y,
        'div': x / (y + 1e-8),
        'pow': x ** 2,

        # Reductions
        'sum_all': x.sum(),
        'sum_axis0': x.sum(dim=0),
        'sum_axis1': x.sum(dim=1),
        'mean': x.mean(),
        'std': x.std(unbiased=False),
        'max': x.max(),
        'argmax': x.argmax(),

        # Shape ops
        'reshape': x.reshape(3, 2),
        'flatten': x.flatten(),
        'unsqueeze': x.unsqueeze(0),            # (1, 2, 3)
        'squeeze': x.unsqueeze(0).squeeze(0),   # back to (2, 3)
        'transpose': x.T,
    }
`,testCases:[{id:"1",description:"Addition",input:'tensor_ops(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))["add"]',expected:"[[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]]",hidden:!1},{id:"2",description:"Mean computation",input:'tensor_ops(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))["mean"]',expected:"3.5",hidden:!1},{id:"3",description:"Reductions along dims, std and argmax",input:'(lambda r: (r["sum_axis0"].tolist(), r["sum_axis1"].tolist(), round(float(r["std"]), 4), int(r["argmax"])))(tensor_ops(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])))',expected:"([5.0, 7.0, 9.0], [6.0, 15.0], 1.7078, 5)",hidden:!0},{id:"4",description:"Shape operations",input:'(lambda r: (tuple(r["reshape"].shape), tuple(r["flatten"].shape), tuple(r["unsqueeze"].shape), tuple(r["squeeze"].shape), tuple(r["transpose"].shape)))(tensor_ops(torch.ones(2, 3), torch.ones(2, 3)))',expected:"((3, 2), (6,), (1, 2, 3), (2, 3), (3, 2))",hidden:!0},{id:"5",description:"Reshape and division values",input:'(lambda r: (r["reshape"].tolist(), r["div"].round().tolist()))(tensor_ops(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])))',expected:"([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])",hidden:!0}],hints:["Element-wise ops work directly: x + y, x * y","Reductions take dim=: x.sum(dim=0)","x.unsqueeze(0) adds a leading dimension; .squeeze(0) removes it"]},"autograd-concepts":{note:"Let autograd compute the gradients: create tensors with requires_grad=True, call .backward(), then read .grad.",starterCode:`import torch

def compute_gradients(x: float) -> dict:
    """
    Compute forward and backward pass for y = x^2 + 3x + 1 using autograd.

    Args:
        x: Input value

    Returns:
        Dictionary with 'y' (forward result) and 'grad' (dy/dx) as Python floats
    """
    # Your code here
    pass


def linear_layer_gradients(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor,
                           grad_output: torch.Tensor) -> dict:
    """
    Compute gradients for a linear layer: Y = X @ W + b, using autograd.

    Args:
        X: Input (batch, in_features)
        W: Weights (in_features, out_features)
        b: Bias (out_features,)
        grad_output: Gradient from next layer (batch, out_features)

    Returns:
        Dictionary with gradients 'grad_X', 'grad_W' and 'grad_b'
    """
    # Your code here
    pass
`,solution:`import torch

def compute_gradients(x: float) -> dict:
    x = torch.tensor(float(x), requires_grad=True)

    # Forward pass builds the graph
    y = x ** 2 + 3 * x + 1

    # Backward pass fills x.grad with dy/dx
    y.backward()

    return {'y': y.item(), 'grad': x.grad.item()}


def linear_layer_gradients(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor,
                           grad_output: torch.Tensor) -> dict:
    # Fresh leaf tensors that track gradients
    X = X.detach().clone().requires_grad_(True)
    W = W.detach().clone().requires_grad_(True)
    b = b.detach().clone().requires_grad_(True)

    # Forward
    Y = X @ W + b

    # Backward: Y is not a scalar, so pass the upstream gradient dL/dY
    Y.backward(grad_output)

    return {
        'grad_W': W.grad,
        'grad_b': b.grad,
        'grad_X': X.grad,
    }
`,testCases:[{id:"1",description:"Simple gradient",input:'compute_gradients(2.0)["grad"]',expected:"7.0",hidden:!1},{id:"2",description:"Forward pass result",input:'compute_gradients(2.0)["y"]',expected:"11.0",hidden:!1},{id:"3",description:"Linear layer: grad_W = X^T @ grad_output",input:'linear_layer_gradients(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]), torch.zeros(3), torch.ones(2, 3))["grad_W"]',expected:"[[4.0, 4.0, 4.0], [6.0, 6.0, 6.0]]",hidden:!1},{id:"4",description:"Gradient at another point",input:'(lambda r: (r["y"], r["grad"]))(compute_gradients(-1.5))',expected:"(-1.25, 0.0)",hidden:!0},{id:"5",description:"Linear layer: grad_b and grad_X",input:'(lambda r: (r["grad_b"].tolist(), r["grad_X"].tolist()))(linear_layer_gradients(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]), torch.zeros(3), torch.ones(2, 3)))',expected:"([2.0, 2.0, 2.0], [[0.0, 3.0], [0.0, 3.0]])",hidden:!0},{id:"6",description:"Linear layer gradients match the analytic formulas",input:`(lambda: (
    X := torch.randn(4, 3),
    W := torch.randn(3, 2),
    b := torch.randn(2),
    g := torch.randn(4, 2),
    r := linear_layer_gradients(X, W, b, g),
    bool(torch.allclose(r["grad_W"], X.T @ g, atol=1e-5)
         and torch.allclose(r["grad_X"], g @ W.T, atol=1e-5)
         and torch.allclose(r["grad_b"], g.sum(dim=0), atol=1e-5))
)[-1])()`,expected:"True",hidden:!0}],hints:["torch.tensor(2.0, requires_grad=True) creates a leaf tensor that records operations","Call y.backward() on a scalar; the gradient lands in x.grad","For a non-scalar output Y, call Y.backward(grad_output)","Use .item() to turn a 0-d tensor into a Python float"]},"nn-modules":{note:"Subclass nn.Module and register weights as nn.Parameter so parameters() finds them. Don't just wrap nn.Linear or nn.Sequential.",starterCode:`import torch
import torch.nn as nn

class Linear(nn.Module):
    """Linear layer: Y = X @ W + b"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Create self.W (in_features, out_features) and self.b (out_features,)
        # as nn.Parameter, with small random weights and zero bias
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass


class ReLU(nn.Module):
    """ReLU activation: max(0, x)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass


class Sequential(nn.Module):
    """Container for sequential layers"""

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)  # registers the sub-modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your code here
        pass


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    """Create a 2-layer MLP: Linear -> ReLU -> Linear"""
    # Your code here
    pass
`,solution:`import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W + self.b

    # parameters() is inherited from nn.Module: it yields W and b


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0)


class Sequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)  # calling the module runs its forward()
        return x


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    return Sequential(
        Linear(input_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, output_dim),
    )
`,testCases:[{id:"1",description:"Linear output shape",input:"tuple(Linear(10, 5).forward(torch.randn(4, 10)).shape)",expected:"(4, 5)",hidden:!1},{id:"2",description:"Sequential forward",input:"tuple(create_mlp(10, 20, 5)(torch.randn(4, 10)).shape)",expected:"(4, 5)",hidden:!1},{id:"3",description:"ReLU values",input:"ReLU()(torch.tensor([-2.0, 0.0, 1.5])).tolist()",expected:"[0.0, 0.0, 1.5]",hidden:!1},{id:"4",description:"Parameters are registered (10*20 + 20 + 20*5 + 5)",input:"sum(p.numel() for p in create_mlp(10, 20, 5).parameters())",expected:"325",hidden:!0},{id:"5",description:"Linear computes x @ W + b with zero bias at init",input:`(lambda: (
    layer := Linear(3, 2),
    x := torch.randn(4, 3),
    bool(torch.allclose(layer(x), x @ layer.W + layer.b) and torch.equal(layer.b.detach(), torch.zeros(2)))
)[-1])()`,expected:"True",hidden:!0},{id:"6",description:"Gradients reach every parameter",input:`(lambda: (
    model := create_mlp(4, 8, 2),
    model(torch.randn(3, 4)).sum().backward(),
    all(p.grad is not None for p in model.parameters())
)[-1])()`,expected:"True",hidden:!0}],hints:["self.W = nn.Parameter(torch.randn(in_features, out_features) * 0.01)","ReLU: torch.clamp(x, min=0)","Sequential.forward applies each layer in order: x = layer(x)","nn.Module.parameters() finds every nn.Parameter, including those in sub-modules"]},"loss-functions":{note:"This is the PyTorch Basics section: use the built-in losses (F.softmax, F.cross_entropy, F.mse_loss, F.binary_cross_entropy). Return Python floats with .item().",starterCode:`import torch
import torch.nn.functional as F

def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension."""
    # Your code here
    pass


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Cross-entropy loss for classification.

    Args:
        logits: Raw scores (batch, num_classes)
        targets: Integer class labels (batch,)

    Returns:
        Scalar loss
    """
    # Your code here
    pass


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Mean squared error loss."""
    # Your code here
    pass


def binary_cross_entropy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Binary cross-entropy on probabilities."""
    # Your code here
    pass
`,solution:`import torch
import torch.nn.functional as F

def softmax(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=-1)


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # F.cross_entropy = log_softmax + negative log-likelihood, averaged over the batch
    return F.cross_entropy(logits, targets).item()


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return F.mse_loss(y_pred, y_true).item()


def binary_cross_entropy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return F.binary_cross_entropy(y_pred, y_true).item()
`,testCases:[{id:"1",description:"Cross-entropy computation",input:"round(float(cross_entropy_loss(torch.tensor([[2.0, 1.0, 0.1]]), torch.tensor([0]))), 4)",expected:"0.417",hidden:!1},{id:"2",description:"MSE computation",input:"float(mse_loss(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 4.0])))",expected:"0.333333",hidden:!1},{id:"3",description:"Softmax values",input:"bool(torch.allclose(softmax(torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])), torch.tensor([[0.090031, 0.244728, 0.665241], [1 / 3, 1 / 3, 1 / 3]]), atol=1e-5))",expected:"True",hidden:!0},{id:"4",description:"Cross-entropy averaged over a batch",input:"round(float(cross_entropy_loss(torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), torch.tensor([2, 0]))), 4)",expected:"1.4076",hidden:!0},{id:"5",description:"Binary cross-entropy",input:"round(float(binary_cross_entropy(torch.tensor([0.9, 0.1, 0.8]), torch.tensor([1.0, 0.0, 1.0]))), 4)",expected:"0.1446",hidden:!0}],hints:["F.softmax(x, dim=-1) normalizes over the last dimension","F.cross_entropy takes raw logits and integer class indices (it applies log_softmax for you)","F.binary_cross_entropy takes probabilities; F.binary_cross_entropy_with_logits takes raw scores","Call .item() on a 0-d tensor to get a Python float"]}},qx={"normalize-features":{note:"Use tensor reductions (arr.min(), arr.max()) and broadcasting.",starterCode:`import torch

def normalize(arr: torch.Tensor) -> torch.Tensor:
    """
    Apply min-max normalization to scale values to [0, 1].

    Args:
        arr: Input tensor of numbers

    Returns:
        Normalized tensor with values in [0, 1]
    """
    # Your code here
    pass
`,solution:`import torch

def normalize(arr: torch.Tensor) -> torch.Tensor:
    """
    Apply min-max normalization to scale values to [0, 1].
    """
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)
`,testCases:[{id:"1",description:"Basic normalization",input:"normalize(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))",expected:"[0.0, 0.25, 0.5, 0.75, 1.0]",hidden:!1},{id:"2",description:"With negative values",input:"normalize(torch.tensor([-10.0, 0.0, 10.0]))",expected:"[0.0, 0.5, 1.0]",hidden:!1},{id:"3",description:"Larger range",input:"normalize(torch.tensor([0.0, 50.0, 100.0]))",expected:"[0.0, 0.5, 1.0]",hidden:!0},{id:"4",description:"Unsorted input",input:"normalize(torch.tensor([3.0, -1.0, 7.0, 1.0]))",expected:"[0.5, 0.0, 1.0, 0.25]",hidden:!0}],hints:["arr.min() and arr.max() return 0-d tensors that broadcast against arr.","Apply the formula: (x - min) / (max - min)"]},"handle-missing-data":{note:"Use torch.isnan and boolean-mask indexing. Clone the input so it is not modified in place.",starterCode:`import torch

def fill_missing_with_mean(arr: torch.Tensor) -> torch.Tensor:
    """
    Replace NaN values with the mean of non-NaN values.

    Args:
        arr: Float tensor that may contain NaN values

    Returns:
        Tensor with NaN values replaced by the mean
    """
    # Your code here
    pass
`,solution:`import torch

def fill_missing_with_mean(arr: torch.Tensor) -> torch.Tensor:
    """
    Replace NaN values with the mean of non-NaN values.
    """
    mask = torch.isnan(arr)
    mean_val = arr[~mask].mean()
    out = arr.clone()  # don't modify the original
    out[mask] = mean_val
    return out
`,testCases:[{id:"1",description:"Basic case",input:'fill_missing_with_mean(torch.tensor([1.0, float("nan"), 3.0, float("nan"), 5.0]))',expected:"[1.0, 3.0, 3.0, 3.0, 5.0]",hidden:!1},{id:"2",description:"Single NaN",input:'fill_missing_with_mean(torch.tensor([2.0, 4.0, float("nan"), 6.0]))',expected:"[2.0, 4.0, 4.0, 6.0]",hidden:!1},{id:"3",description:"NaN at start",input:'fill_missing_with_mean(torch.tensor([float("nan"), 10.0, 20.0]))',expected:"[15.0, 10.0, 20.0]",hidden:!0},{id:"4",description:"No NaN: unchanged",input:"fill_missing_with_mean(torch.tensor([1.5, -2.0, 0.5]))",expected:"[1.5, -2.0, 0.5]",hidden:!0}],hints:["torch.isnan(arr) gives a boolean mask of the missing values.","arr[~mask].mean() is the mean of the values that are present.","out[mask] = value assigns through a boolean mask."]},"one-hot-encode":{note:"F.one_hot is the built-in; indexing torch.eye(num_classes) with the labels also works. Return an integer tensor.",starterCode:`import torch
import torch.nn.functional as F

def one_hot_encode(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer labels to one-hot encoded format.

    Args:
        labels: 1D int64 tensor of labels
        num_classes: Total number of classes

    Returns:
        2D integer tensor of shape (len(labels), num_classes)
    """
    # Your code here
    pass
`,solution:`import torch
import torch.nn.functional as F

def one_hot_encode(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer labels to one-hot encoded format.
    """
    # Method 1: built-in
    return F.one_hot(labels, num_classes)

    # Method 2: index rows of the identity matrix
    # return torch.eye(num_classes)[labels].long()
`,testCases:[{id:"1",description:"Basic encoding",input:"one_hot_encode(torch.tensor([0, 1, 2]), 3)",expected:"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",hidden:!1},{id:"2",description:"Repeated labels",input:"one_hot_encode(torch.tensor([0, 0, 1, 1]), 2)",expected:"[[1, 0], [1, 0], [0, 1], [0, 1]]",hidden:!1},{id:"3",description:"More classes than used",input:"one_hot_encode(torch.tensor([0, 2]), 4)",expected:"[[1, 0, 0, 0], [0, 0, 1, 0]]",hidden:!0},{id:"4",description:"Each row sums to 1",input:"one_hot_encode(torch.tensor([3, 1, 4, 1, 0]), 5).sum(dim=1).tolist()",expected:"[1, 1, 1, 1, 1]",hidden:!0}],hints:["F.one_hot(labels, num_classes) does it in one call.","Or: torch.eye(num_classes)[labels] picks one identity row per label.","Convert to integers with .long() if you build it from a float tensor."]}},Ox={"logistic-regression":{note:"Implement it with torch.exp; don't call torch.sigmoid or F.sigmoid.",starterCode:`import torch

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the sigmoid activation function.

    Args:
        x: Float tensor of any shape

    Returns:
        Tensor of same shape with sigmoid applied element-wise
    """
    # Your code here
    pass
`,solution:`import torch

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the sigmoid activation function.
    """
    return 1 / (1 + torch.exp(-x))
`,testCases:[{id:"1",description:"Zero input",input:"sigmoid(torch.tensor([0.0]))",expected:"[0.5]",hidden:!1},{id:"2",description:"Positive and negative",input:"bool(torch.allclose(sigmoid(torch.tensor([-1.0, 0.0, 1.0])), torch.tensor([0.268941, 0.5, 0.731059]), atol=1e-5))",expected:"True",hidden:!1},{id:"3",description:"Large values",input:"bool(torch.allclose(sigmoid(torch.tensor([-10.0, 10.0])), torch.tensor([4.5e-05, 0.999955]), atol=1e-5))",expected:"True",hidden:!0},{id:"4",description:"Matches torch.sigmoid on a 2D input",input:"(lambda x: bool(torch.allclose(sigmoid(x), torch.sigmoid(x), atol=1e-6)))(torch.randn(3, 4))",expected:"True",hidden:!0}],hints:["Use torch.exp() for the exponential function.","The formula is 1 / (1 + exp(-x)); it broadcasts over any shape."]},"binary-cross-entropy":{note:"Implement the formula with tensor ops (F.binary_cross_entropy is the built-in). In float32, 1 - 1e-15 rounds to 1.0, so clamp with eps = 1e-7.",starterCode:`import torch

def binary_cross_entropy(y_true, y_pred):
    """
    Compute binary cross-entropy loss.

    Args:
        y_true: True labels (0 or 1), tensor
        y_pred: Predicted probabilities, tensor

    Returns:
        loss: Scalar BCE loss (Python float rounded to 4 decimals)
    """
    # Your code here
    pass
`,solution:`import torch

def binary_cross_entropy(y_true, y_pred):
    y_true = y_true.float()
    eps = 1e-7  # 1e-15 is below float32 resolution near 1.0

    # Clamp for numerical stability
    y_pred = torch.clamp(y_pred, eps, 1 - eps)

    # BCE formula
    loss = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

    return round(loss.item(), 4)
`,testCases:[{id:"1",description:"Perfect predictions",input:"binary_cross_entropy(torch.tensor([1.0, 0.0, 1.0, 0.0]), torch.tensor([1.0, 0.0, 1.0, 0.0]))",expected:"0.0",hidden:!1},{id:"2",description:"Typical case",input:"binary_cross_entropy(torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.9, 0.1, 0.8]))",expected:"0.1446",hidden:!1},{id:"3",description:"Less confident predictions",input:"binary_cross_entropy(torch.tensor([1.0, 1.0, 0.0]), torch.tensor([0.7, 0.5, 0.2]))",expected:"0.4243",hidden:!0},{id:"4",description:"Integer labels",input:"binary_cross_entropy(torch.tensor([0, 1]), torch.tensor([0.25, 0.75]))",expected:"0.2877",hidden:!0}],hints:["Clamp predictions with torch.clamp(y_pred, eps, 1 - eps)","Apply the formula element-wise with torch.log","Take the mean, then .item() and round to 4 decimals"]},"hinge-loss":{note:"Use tensor ops: torch.clamp(..., min=0) is max(0, ...).",starterCode:`import torch

def hinge_loss(y_true, scores):
    """
    Compute the mean hinge loss.

    Args:
        y_true: True labels in {-1, +1}, tensor of shape (n,)
        scores: Raw model scores, tensor of shape (n,)

    Returns:
        Mean hinge loss (float, rounded to 4 decimals)
    """
    # Your code here
    pass
`,solution:`import torch

def hinge_loss(y_true, scores):
    """
    Compute the mean hinge loss.
    """
    # max(0, 1 - y * s) for every sample
    losses = torch.clamp(1 - y_true * scores, min=0)

    return round(losses.mean().item(), 4)
`,testCases:[{id:"1",description:"Perfect predictions with large margin",input:"hinge_loss(torch.tensor([1.0, -1.0]), torch.tensor([2.0, -3.0]))",expected:"0.0",hidden:!1},{id:"2",description:"Partial margin violations",input:"hinge_loss(torch.tensor([1.0, -1.0, 1.0]), torch.tensor([0.5, -0.8, 1.5]))",expected:"0.2333",hidden:!1},{id:"3",description:"All misclassified",input:"hinge_loss(torch.tensor([1.0, 1.0]), torch.tensor([-1.0, -2.0]))",expected:"2.5",hidden:!1},{id:"4",description:"On the decision boundary (margin = 1)",input:"hinge_loss(torch.tensor([1.0, -1.0]), torch.tensor([1.0, -1.0]))",expected:"0.0",hidden:!0},{id:"5",description:"Integer labels",input:"hinge_loss(torch.tensor([1, -1, -1, 1]), torch.tensor([0.25, 0.5, -2.0, 3.0]))",expected:"0.5625",hidden:!0}],hints:["Compute element-wise margins: y_true * scores","Apply max(0, 1 - margin) using torch.clamp(..., min=0)","Return the mean of the losses with .item(), rounded to 4 decimals"]},"decision-tree-split":{note:"torch.bincount counts each integer label in one call (classes that never appear count 0 and add nothing to the sum).",starterCode:`import torch

def gini_impurity(labels: torch.Tensor) -> float:
    """
    Calculate Gini impurity for a set of labels.

    Args:
        labels: 1D int64 tensor of class labels (non-negative integers)

    Returns:
        Gini impurity value between 0 and 1 (Python float)
    """
    # Your code here
    pass
`,solution:`import torch

def gini_impurity(labels: torch.Tensor) -> float:
    """
    Calculate Gini impurity for a set of labels.
    """
    if labels.numel() == 0:
        return 0.0

    # Count occurrences of each class
    counts = torch.bincount(labels)

    # Class probabilities
    probabilities = counts.float() / labels.numel()

    # Gini = 1 - sum(p^2)
    return (1 - (probabilities ** 2).sum()).item()
`,testCases:[{id:"1",description:"Pure node",input:"gini_impurity(torch.tensor([0, 0, 0, 0]))",expected:"0.0",hidden:!1},{id:"2",description:"Maximum impurity",input:"gini_impurity(torch.tensor([0, 0, 1, 1]))",expected:"0.5",hidden:!1},{id:"3",description:"Unequal split",input:"gini_impurity(torch.tensor([0, 0, 0, 1]))",expected:"0.375",hidden:!0},{id:"4",description:"Three classes",input:"gini_impurity(torch.tensor([2, 0, 1, 2]))",expected:"0.625",hidden:!0}],hints:["torch.bincount(labels) counts each class.","Divide by labels.numel() to get proportions (convert to float first).","Gini = 1 - sum of squared probabilities; return it with .item()."]},"linear-regression-gd":{note:"Let autograd compute dw and db: build the MSE loss, call loss.backward(), update under torch.no_grad(), then zero the gradients.",starterCode:`import torch

def linear_regression(X: torch.Tensor, y: torch.Tensor,
                      learning_rate: float = 0.01,
                      iterations: int = 1000) -> tuple:
    """
    Train a simple linear regression model using gradient descent.

    Args:
        X: Input features (1D float tensor)
        y: Target values (1D float tensor)
        learning_rate: Step size for gradient descent
        iterations: Number of training iterations

    Returns:
        Tuple of (weight, bias) as Python floats rounded to 2 decimals
    """
    # Initialize parameters (autograd tracks them)
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # Your gradient descent implementation here
    pass

    return (round(w.item(), 2), round(b.item(), 2))
`,solution:`import torch

def linear_regression(X: torch.Tensor, y: torch.Tensor,
                      learning_rate: float = 0.01,
                      iterations: int = 1000) -> tuple:
    """
    Train a simple linear regression model using gradient descent.
    """
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    for _ in range(iterations):
        # Forward pass and MSE loss
        y_pred = w * X + b
        loss = ((y_pred - y) ** 2).mean()

        # Backward pass: w.grad = (2/n) * sum((y_pred - y) * X), b.grad = (2/n) * sum(y_pred - y)
        loss.backward()

        # Update parameters without recording the update in the graph
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            w.grad.zero_()
            b.grad.zero_()

    return (round(w.item(), 2), round(b.item(), 2))
`,testCases:[{id:"1",description:"Perfect linear relationship",input:"linear_regression(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([2.0, 4.0, 6.0, 8.0]), 0.1, 1000)",expected:"(2.0, 0.0)",hidden:!1},{id:"2",description:"With intercept",input:"linear_regression(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([3.0, 5.0, 7.0, 9.0]), 0.1, 1000)",expected:"(2.0, 1.0)",hidden:!1},{id:"3",description:"Different slope",input:"linear_regression(torch.tensor([0.0, 1.0, 2.0, 3.0]), torch.tensor([1.0, 4.0, 7.0, 10.0]), 0.1, 1000)",expected:"(3.0, 1.0)",hidden:!0},{id:"4",description:"Noisy data converges to the least-squares fit",input:"linear_regression(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]), torch.tensor([1.1, 2.9, 5.2, 7.1, 8.8]), 0.05, 2000)",expected:"(1.96, 1.1)",hidden:!0},{id:"5",description:"Few iterations (checks the update rule, not just the optimum)",input:"linear_regression(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([2.0, 4.0, 6.0, 8.0]), 0.01, 10)",expected:"(1.51, 0.49)",hidden:!0}],hints:["Forward: y_pred = w * X + b, loss = ((y_pred - y) ** 2).mean()","loss.backward() fills w.grad and b.grad","Inside torch.no_grad(): w -= learning_rate * w.grad, then w.grad.zero_() (gradients accumulate otherwise)"]},"logistic-regression-full":{note:"Let autograd compute the gradients of the mean BCE loss (F.binary_cross_entropy_with_logits is fine for the loss); update under torch.no_grad().",starterCode:`import torch
import torch.nn.functional as F

def logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    """
    Train logistic regression using gradient descent.

    Args:
        X: Float features (m samples, n features)
        y: Binary float labels (m,)
        learning_rate: Step size
        iterations: Number of iterations

    Returns:
        w: Learned weights (n,), rounded to 4 decimals
        b: Learned bias (Python float), rounded to 4 decimals
    """
    m, n = X.shape
    w = torch.zeros(n, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # Your code here
    pass

    return torch.round(w.detach(), decimals=4), round(b.item(), 4)
`,solution:`import torch
import torch.nn.functional as F

def logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    m, n = X.shape
    w = torch.zeros(n, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    for _ in range(iterations):
        # Forward pass: logits, then mean binary cross-entropy
        z = X @ w + b
        loss = F.binary_cross_entropy_with_logits(z, y)

        # Backward: w.grad = (1/m) X^T (sigmoid(z) - y), b.grad = mean(sigmoid(z) - y)
        loss.backward()

        # Update parameters
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            w.grad.zero_()
            b.grad.zero_()

    return torch.round(w.detach(), decimals=4), round(b.item(), 4)
`,testCases:[{id:"1",description:"Simple separable data (AND)",input:"(lambda r: bool(torch.allclose(r[0], torch.tensor([6.0141, 6.0141]), atol=1e-3) and abs(r[1] - (-9.1984)) < 1e-3))(logistic_regression(torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), torch.tensor([0.0, 0.0, 0.0, 1.0]), 0.5, 1000))",expected:"True",hidden:!1},{id:"2",description:"OR data",input:"(lambda r: bool(torch.allclose(r[0], torch.tensor([7.2375, 7.2375]), atol=1e-3) and abs(r[1] - (-3.1461)) < 1e-3))(logistic_regression(torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), torch.tensor([0.0, 1.0, 1.0, 1.0]), 0.5, 1000))",expected:"True",hidden:!0},{id:"3",description:"Two clusters, fewer iterations",input:"(lambda r: bool(torch.allclose(r[0], torch.tensor([0.988, 0.0689]), atol=1e-3) and abs(r[1] - (-4.2202)) < 1e-3))(logistic_regression(torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [6.0, 5.0], [7.0, 8.0], [8.0, 6.0]]), torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 0.1, 500))",expected:"True",hidden:!0}],hints:["Forward: z = X @ w + b; loss = F.binary_cross_entropy_with_logits(z, y) (mean over samples)","loss.backward() gives exactly dw = (1/m) X^T (y_pred - y) and db = mean(y_pred - y)","Update inside torch.no_grad(), then zero w.grad and b.grad"]},"linear-svm":{note:"Write the objective (λ/2)·||w||² + mean(relu(1 - margins)) and let autograd differentiate it. torch.relu has gradient 0 at the hinge (margin exactly 1), matching the strict margin < 1 rule.",starterCode:`import torch

def linear_svm(X, y, learning_rate=0.01, lambda_param=0.01, iterations=1000):
    """
    Train a linear SVM using gradient descent on hinge loss + L2 regularization.

    Args:
        X: Float features (m samples, n features)
        y: Float labels in {-1, +1}, shape (m,)
        learning_rate: Step size for gradient descent
        lambda_param: L2 regularization strength
        iterations: Number of training iterations

    Returns:
        w: Learned weights (n,), rounded to 4 decimals
        b: Learned bias (Python float), rounded to 4 decimals
    """
    m, n = X.shape
    w = torch.zeros(n, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # Your gradient descent implementation here
    pass

    return torch.round(w.detach(), decimals=4), round(b.item(), 4)
`,solution:`import torch

def linear_svm(X, y, learning_rate=0.01, lambda_param=0.01, iterations=1000):
    """
    Train a linear SVM using gradient descent on hinge loss + L2 regularization.
    """
    m, n = X.shape
    w = torch.zeros(n, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    for _ in range(iterations):
        # Margins and objective
        margins = y * (X @ w + b)
        loss = lambda_param / 2 * (w ** 2).sum() + torch.relu(1 - margins).mean()

        # Autograd: only samples with margin < 1 contribute to the hinge gradient
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            w.grad.zero_()
            b.grad.zero_()

    return torch.round(w.detach(), decimals=4), round(b.item(), 4)
`,testCases:[{id:"1",description:"Correctly classifies separable 2D data",input:`(lambda: (
    X := torch.tensor([[2.0, 1.0], [3.0, 2.0], [1.0, 3.0], [-2.0, -1.0], [-3.0, -2.0], [-1.0, -3.0]]),
    y := torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]),
    result := linear_svm(X, y, 0.01, 0.01, 1000),
    preds := torch.sign(X @ result[0] + result[1]),
    bool(torch.equal(preds, y))
)[-1])()`,expected:"True",hidden:!1},{id:"2",description:"Correctly classifies 1D data",input:`(lambda: (
    X := torch.tensor([[3.0], [4.0], [5.0], [-3.0], [-4.0], [-5.0]]),
    y := torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]),
    result := linear_svm(X, y, 0.01, 0.01, 1000),
    preds := torch.sign(X @ result[0] + result[1]),
    bool(torch.equal(preds, y))
)[-1])()`,expected:"True",hidden:!1},{id:"3",description:"Weight vector has correct sign",input:`(lambda: (
    X := torch.tensor([[1.0], [-1.0]]),
    y := torch.tensor([1.0, -1.0]),
    result := linear_svm(X, y, 0.1, 0.01, 500),
    bool(result[0][0] > 0)
)[-1])()`,expected:"True",hidden:!0},{id:"4",description:"Learned weights and bias match gradient descent",input:`(lambda: (
    X := torch.tensor([[2.0, 0.5], [1.0, 1.5], [-1.0, -0.5], [-0.5, -2.0]]),
    y := torch.tensor([1.0, 1.0, -1.0, -1.0]),
    result := linear_svm(X, y, 0.05, 0.1, 200),
    bool(torch.allclose(result[0], torch.tensor([0.5321, 0.4822]), atol=1e-3) and abs(result[1] - (-0.2375)) < 1e-3)
)[-1])()`,expected:"True",hidden:!0}],hints:["margins = y * (X @ w + b)","loss = lambda_param / 2 * (w ** 2).sum() + torch.relu(1 - margins).mean()","loss.backward(), then update w and b inside torch.no_grad() and zero their gradients"]}},Wx={"kmeans-clustering":{note:"Broadcast with unsqueeze to get all point-centroid differences at once, then take the norm over the feature dim and argmin over centroids.",starterCode:`import torch

def assign_clusters(X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Assign each point to its nearest centroid.

    Args:
        X: Float data points of shape (n_samples, n_features)
        centroids: Float cluster centers of shape (k, n_features)

    Returns:
        int64 tensor of cluster assignments (shape: n_samples)
    """
    # Your code here
    pass
`,solution:`import torch

def assign_clusters(X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Assign each point to its nearest centroid.
    """
    # (n, 1, d) - (1, k, d) -> (n, k, d) differences
    diffs = X.unsqueeze(1) - centroids.unsqueeze(0)

    # Euclidean distance from each point to each centroid: (n, k)
    distances = diffs.norm(dim=2)

    # Index of the closest centroid (first one wins ties)
    return distances.argmin(dim=1)
`,testCases:[{id:"1",description:"Simple 2D case",input:"assign_clusters(torch.tensor([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]]), torch.tensor([[0.0, 0.0], [10.0, 10.0]]))",expected:"[0, 0, 1]",hidden:!1},{id:"2",description:"Three clusters",input:"assign_clusters(torch.tensor([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]]), torch.tensor([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]]))",expected:"[0, 1, 2]",hidden:!1},{id:"3",description:"Tiebreaker (first centroid wins)",input:"assign_clusters(torch.tensor([[5.0, 5.0]]), torch.tensor([[0.0, 0.0], [10.0, 10.0]]))",expected:"[0]",hidden:!0},{id:"4",description:"Random points match a brute-force reference",input:`(lambda: (
    X := torch.randn(20, 3),
    C := torch.randn(4, 3),
    ref := ((X[:, None, :] - C[None, :, :]) ** 2).sum(dim=-1).argmin(dim=1),
    bool(torch.equal(assign_clusters(X, C), ref))
)[-1])()`,expected:"True",hidden:!0}],hints:["X.unsqueeze(1) has shape (n, 1, d); centroids.unsqueeze(0) has shape (1, k, d).","diffs.norm(dim=2) gives the (n, k) Euclidean distances.","distances.argmin(dim=1) picks the closest centroid."]},"pca-implementation":{note:"Use X.mean(dim=0) and broadcasting.",starterCode:`import torch

def center_data(X: torch.Tensor) -> torch.Tensor:
    """
    Center the data by subtracting the mean of each feature.

    Args:
        X: Float data matrix of shape (n_samples, n_features)

    Returns:
        Centered data matrix of same shape
    """
    # Your code here
    pass
`,solution:`import torch

def center_data(X: torch.Tensor) -> torch.Tensor:
    """
    Center the data by subtracting the mean of each feature.
    """
    mean = X.mean(dim=0)  # (n_features,)
    return X - mean
`,testCases:[{id:"1",description:"Basic centering",input:"center_data(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))",expected:"[[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]]",hidden:!1},{id:"2",description:"Already centered",input:"center_data(torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]))",expected:"[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]",hidden:!1},{id:"3",description:"Single column",input:"center_data(torch.tensor([[10.0], [20.0], [30.0]]))",expected:"[[-10.0], [0.0], [10.0]]",hidden:!0},{id:"4",description:"Columns of random data have zero mean",input:"bool(torch.allclose(center_data(torch.randn(50, 4) * 3 + 7).mean(dim=0), torch.zeros(4), atol=1e-4))",expected:"True",hidden:!0}],hints:["X.mean(dim=0) gives the mean of each column.","Subtract it from X; broadcasting handles the rows."]}},Vx={"precision-recall-f1":{note:"Build boolean masks with == and &, count with .sum().item(), and return Python floats.",starterCode:`import torch

def precision_recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple:
    """
    Calculate precision and recall for binary classification.

    Args:
        y_true: True labels (0 or 1), int64 tensor
        y_pred: Predicted labels (0 or 1), int64 tensor

    Returns:
        Tuple of (precision, recall) rounded to 4 decimal places
    """
    # Your code here
    pass
`,solution:`import torch

def precision_recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple:
    """
    Calculate precision and recall for binary classification.
    """
    # Calculate TP, FP, FN as Python ints
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return (round(precision, 4), round(recall, 4))
`,testCases:[{id:"1",description:"Mixed predictions",input:"precision_recall(torch.tensor([1, 1, 0, 1, 0]), torch.tensor([1, 0, 0, 1, 1]))",expected:"(0.6667, 0.6667)",hidden:!1},{id:"2",description:"Perfect predictions",input:"precision_recall(torch.tensor([1, 1, 0, 0]), torch.tensor([1, 1, 0, 0]))",expected:"(1.0, 1.0)",hidden:!1},{id:"3",description:"High precision, low recall",input:"precision_recall(torch.tensor([1, 1, 1, 1, 0]), torch.tensor([1, 0, 0, 0, 0]))",expected:"(1.0, 0.25)",hidden:!0},{id:"4",description:"Low precision, full recall",input:"precision_recall(torch.tensor([0, 0, 1, 0, 0, 0, 1]), torch.tensor([1, 1, 1, 0, 1, 0, 1]))",expected:"(0.4, 1.0)",hidden:!0}],hints:["TP: ((y_true == 1) & (y_pred == 1)).sum()","FP: predicted 1 but actually 0; FN: actually 1 but predicted 0","Use .item() to turn the counts into Python ints before dividing."]},"cross-validation":{note:"Slice torch.arange(n_samples) into folds and join the rest with torch.cat. Return plain Python lists (.tolist()).",starterCode:`import torch

def kfold_indices(n_samples: int, k: int) -> list:
    """
    Generate train/validation indices for K-fold cross-validation.

    Args:
        n_samples: Total number of samples
        k: Number of folds

    Returns:
        List of (train_indices, val_indices) tuples of Python lists
    """
    # Your code here
    pass
`,solution:`import torch

def kfold_indices(n_samples: int, k: int) -> list:
    """
    Generate train/validation indices for K-fold cross-validation.
    """
    indices = torch.arange(n_samples)

    # The first n_samples % k folds get one extra sample
    fold_sizes = [n_samples // k + (1 if i < n_samples % k else 0) for i in range(k)]

    folds = []
    start = 0
    for size in fold_sizes:
        val_idx = indices[start:start + size]
        train_idx = torch.cat([indices[:start], indices[start + size:]])
        folds.append((train_idx.tolist(), val_idx.tolist()))
        start += size

    return folds
`,testCases:[{id:"1",description:"Basic 3-fold",input:"kfold_indices(6, 3)",expected:"[[[2, 3, 4, 5], [0, 1]], [[0, 1, 4, 5], [2, 3]], [[0, 1, 2, 3], [4, 5]]]",hidden:!1},{id:"2",description:"2-fold",input:"kfold_indices(4, 2)",expected:"[[[2, 3], [0, 1]], [[0, 1], [2, 3]]]",hidden:!1},{id:"3",description:"4-fold",input:"kfold_indices(8, 4)",expected:"[[[2, 3, 4, 5, 6, 7], [0, 1]], [[0, 1, 4, 5, 6, 7], [2, 3]], [[0, 1, 2, 3, 6, 7], [4, 5]], [[0, 1, 2, 3, 4, 5], [6, 7]]]",hidden:!0},{id:"4",description:"Uneven folds (first fold gets the extra sample)",input:"kfold_indices(5, 2)",expected:"[[[3, 4], [0, 1, 2]], [[0, 1, 2], [3, 4]]]",hidden:!0}],hints:["Fold sizes: n_samples // k, plus one for the first n_samples % k folds.","indices[start:start + size] is the validation fold.","torch.cat([indices[:start], indices[start + size:]]) is the training set."]},"confusion-matrix":{note:"Compare tensors element-wise, convert the boolean result to float, and take the mean. Return a Python float.",starterCode:`import torch

def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate accuracy of predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score between 0 and 1
    """
    # Your code here
    pass
`,solution:`import torch

def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate accuracy of predictions.
    """
    # A bool tensor has no mean(): convert to float first
    return (y_true == y_pred).float().mean().item()
`,testCases:[{id:"1",description:"Basic accuracy",input:"accuracy(torch.tensor([0, 1, 1, 0, 1]), torch.tensor([0, 1, 0, 0, 1]))",expected:"0.8",hidden:!1},{id:"2",description:"Perfect accuracy",input:"accuracy(torch.tensor([1, 1, 0, 0]), torch.tensor([1, 1, 0, 0]))",expected:"1.0",hidden:!1},{id:"3",description:"Zero accuracy",input:"accuracy(torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 0, 0]))",expected:"0.0",hidden:!0},{id:"4",description:"Multi-class labels",input:"accuracy(torch.tensor([2, 0, 1, 2, 1, 0, 2, 1]), torch.tensor([2, 1, 1, 2, 0, 0, 2, 2]))",expected:"0.625",hidden:!0}],hints:["y_true == y_pred gives a boolean tensor.",".float().mean() gives the fraction of matches.",".item() turns the 0-d result into a Python float."]}},Bx={perceptron:{note:"Implement it with torch.clamp or torch.where; don't call torch.relu or F.relu.",starterCode:`import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply ReLU activation function.

    Args:
        x: Input tensor of any shape

    Returns:
        Tensor with ReLU applied element-wise
    """
    # Your code here
    pass
`,solution:`import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply ReLU activation function.
    """
    return torch.clamp(x, min=0)
`,testCases:[{id:"1",description:"Mixed values",input:"relu(torch.tensor([-2, -1, 0, 1, 2]))",expected:"[0, 0, 0, 1, 2]",hidden:!1},{id:"2",description:"All negative",input:"relu(torch.tensor([-5, -3, -1]))",expected:"[0, 0, 0]",hidden:!1},{id:"3",description:"All positive",input:"relu(torch.tensor([1, 2, 3]))",expected:"[1, 2, 3]",hidden:!0},{id:"4",description:"Float 2D input",input:"relu(torch.tensor([[-1.5, 2.5], [0.25, -0.75]]))",expected:"[[0.0, 2.5], [0.25, 0.0]]",hidden:!0},{id:"5",description:"Gradient is 0 for negatives and 1 for positives",input:"(lambda: (x := torch.tensor([-1.0, 2.0, -3.0, 0.5], requires_grad=True), relu(x).sum().backward(), x.grad.tolist())[-1])()",expected:"[0.0, 1.0, 0.0, 1.0]",hidden:!0}],hints:["torch.clamp(x, min=0) sets every negative value to 0.","Alternatively, torch.where(x > 0, x, torch.zeros_like(x))."]},"neural-network-forward":{note:"Use the @ operator and broadcasting for the bias; apply ReLU with torch.clamp (no nn.Linear needed).",starterCode:`import torch

def dense_forward(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute forward pass of a dense layer with ReLU activation.

    Args:
        X: Input of shape (batch_size, input_dim)
        W: Weights of shape (input_dim, output_dim)
        b: Bias of shape (output_dim,)

    Returns:
        Output of shape (batch_size, output_dim)
    """
    # Your code here
    pass
`,solution:`import torch

def dense_forward(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute forward pass of a dense layer with ReLU activation.
    """
    # Linear transformation
    Z = X @ W + b
    # ReLU activation
    return torch.clamp(Z, min=0)
`,testCases:[{id:"1",description:"Basic forward pass",input:"dense_forward(torch.tensor([[1, 2]]), torch.tensor([[1, 0], [0, 1]]), torch.tensor([1, 1]))",expected:"[[2, 3]]",hidden:!1},{id:"2",description:"With negative pre-activation",input:"dense_forward(torch.tensor([[1, 1]]), torch.tensor([[1, -2], [-1, 1]]), torch.tensor([0, 0]))",expected:"[[0, 0]]",hidden:!1},{id:"3",description:"Batch of 2",input:"dense_forward(torch.tensor([[1, 0], [0, 1]]), torch.tensor([[2, 0], [0, 2]]), torch.tensor([0, 0]))",expected:"[[2, 0], [0, 2]]",hidden:!0},{id:"4",description:"Random float inputs match relu(X @ W + b)",input:`(lambda: (
    X := torch.randn(4, 3),
    W := torch.randn(3, 5),
    b := torch.randn(5),
    bool(torch.allclose(dense_forward(X, W, b), torch.relu(X @ W + b), atol=1e-6))
)[-1])()`,expected:"True",hidden:!0}],hints:["First compute the linear transformation: Z = X @ W + b","Then apply ReLU: torch.clamp(Z, min=0)"]},backpropagation:{note:"Implement it with torch.exp and .sum(); don't call F.softmax or torch.softmax.",starterCode:`import torch

def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax probabilities.

    Args:
        x: Float input tensor (1D)

    Returns:
        Tensor of same shape with softmax probabilities
    """
    # Your code here
    # Hint: subtract max for numerical stability
    pass
`,solution:`import torch

def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax probabilities.
    """
    # Subtract max for numerical stability
    x_shifted = x - x.max()
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum()
`,testCases:[{id:"1",description:"Basic softmax",input:"bool(torch.allclose(softmax(torch.tensor([1.0, 2.0, 3.0])), torch.tensor([0.090031, 0.244728, 0.665241]), atol=1e-5))",expected:"True",hidden:!1},{id:"2",description:"Equal inputs",input:"bool(torch.allclose(softmax(torch.tensor([1.0, 1.0, 1.0])), torch.tensor([0.333333, 0.333333, 0.333333]), atol=1e-5))",expected:"True",hidden:!1},{id:"3",description:"Large values",input:"bool(torch.allclose(softmax(torch.tensor([100.0, 101.0, 102.0])), torch.tensor([0.090031, 0.244728, 0.665241]), atol=1e-5))",expected:"True",hidden:!0},{id:"4",description:"Very large values stay finite",input:"softmax(torch.tensor([1000.0, 1000.0, 1000.0, 1000.0])).tolist()",expected:"[0.25, 0.25, 0.25, 0.25]",hidden:!0}],hints:["Subtract x.max() for numerical stability.","Compute torch.exp of each element.","Divide by the sum of all exp values."]},"mlp-forward-backward":{note:"Write the forward pass with tensor ops and derive the backward pass by hand. Don't call loss.backward() inside mlp_backward. A hidden test checks your gradients against autograd.",starterCode:`import torch

def relu(x):
    return torch.clamp(x, min=0)

def softmax(x):
    exp_x = torch.exp(x - x.max(dim=1, keepdim=True)[0])
    return exp_x / exp_x.sum(dim=1, keepdim=True)

def mlp_forward(X, W1, b1, W2, b2):
    """
    Forward pass for a 2-layer MLP.

    Args:
        X: Input data (batch_size, n_features)
        W1: First layer weights (n_features, n_hidden)
        b1: First layer bias (n_hidden,)
        W2: Second layer weights (n_hidden, n_classes)
        b2: Second layer bias (n_classes,)

    Returns:
        output: Class probabilities (batch_size, n_classes)
        cache: Tuple (X, Z1, A1, Z2, W1, W2) for backprop
    """
    # Your code here
    pass

def mlp_backward(Y, output, cache):
    """
    Backward pass (backpropagation) for 2-layer MLP, derived by hand.

    Args:
        Y: One-hot encoded labels (batch_size, n_classes)
        output: Predicted probabilities from forward pass
        cache: Cached values from forward pass (X, Z1, A1, Z2, W1, W2)

    Returns:
        grads: Dictionary with dW1, db1, dW2, db2
    """
    X, Z1, A1, Z2, W1, W2 = cache
    m = X.shape[0]  # batch size

    # Your code here
    pass
`,solution:`import torch

def relu(x):
    return torch.clamp(x, min=0)

def softmax(x):
    exp_x = torch.exp(x - x.max(dim=1, keepdim=True)[0])
    return exp_x / exp_x.sum(dim=1, keepdim=True)

def mlp_forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    output = softmax(Z2)
    cache = (X, Z1, A1, Z2, W1, W2)
    return output, cache

def mlp_backward(Y, output, cache):
    X, Z1, A1, Z2, W1, W2 = cache
    m = X.shape[0]

    # Softmax + cross-entropy: dL/dZ2 = output - Y
    dZ2 = output - Y
    dW2 = A1.T @ dZ2 / m
    db2 = dZ2.sum(dim=0) / m

    # Through W2 and the ReLU
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0).float()
    dW1 = X.T @ dZ1 / m
    db1 = dZ1.sum(dim=0) / m

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
`,testCases:[{id:"1",description:"Forward: output shape correct",input:"mlp_forward(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.ones(3, 4), torch.zeros(4), torch.ones(4, 2), torch.zeros(2))[0].shape",expected:"(2, 2)",hidden:!1},{id:"2",description:"Forward: identity weights give softmax of the input",input:"bool(torch.allclose(mlp_forward(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.eye(2), torch.zeros(2), torch.eye(2), torch.zeros(2))[0], torch.tensor([[0.7311, 0.2689], [0.2689, 0.7311]]), atol=1e-4))",expected:"True",hidden:!1},{id:"3",description:"Backward: all gradient keys present with the right shapes",input:`(lambda: (
            X := torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            W1 := torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            W2 := torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            result := mlp_forward(X, W1, torch.zeros(3), W2, torch.zeros(2)),
            grads := mlp_backward(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), result[0], result[1]),
            (sorted(grads.keys()), tuple(grads['dW1'].shape), tuple(grads['dW2'].shape))
        )[-1])()`,expected:"(['dW1', 'dW2', 'db1', 'db2'], (2, 3), (3, 2))",hidden:!1},{id:"4",description:"Backward: dW2 matches a hand-computed value",input:`(lambda: (
            X := torch.tensor([[1.0, 2.0]]),
            W1 := torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            W2 := torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            result := mlp_forward(X, W1, torch.zeros(2), W2, torch.zeros(2)),
            grads := mlp_backward(torch.tensor([[1.0, 0.0]]), result[0], result[1]),
            bool(torch.allclose(grads['dW2'], torch.tensor([[-0.5, 0.5], [-1.0, 1.0]]), atol=1e-6))
        )[-1])()`,expected:"True",hidden:!0},{id:"5",description:"Backward: manual gradients match autograd",input:`(lambda: (
            X := torch.tensor([[1.0, 2.0], [3.0, -4.0], [0.5, 0.5]]),
            W1 := torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], requires_grad=True),
            b1 := torch.tensor([0.1, 0.0, -0.1], requires_grad=True),
            W2 := torch.tensor([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6]], requires_grad=True),
            b2 := torch.tensor([0.05, -0.05], requires_grad=True),
            Y := torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            result := mlp_forward(X, W1, b1, W2, b2),
            grads := mlp_backward(Y, result[0], result[1]),
            loss := -(Y * torch.log(result[0])).sum() / 3,
            loss.backward(),
            bool(torch.allclose(grads['dW1'], W1.grad, atol=1e-5) and torch.allclose(grads['db1'], b1.grad, atol=1e-5)
                 and torch.allclose(grads['dW2'], W2.grad, atol=1e-5) and torch.allclose(grads['db2'], b2.grad, atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0}],hints:["Forward: Z1 = X @ W1 + b1, A1 = relu(Z1), Z2 = A1 @ W2 + b2, output = softmax(Z2)","Backward: dZ2 = output - Y; dW2 = A1.T @ dZ2 / m; db2 = dZ2.sum(dim=0) / m","ReLU derivative as a float mask: (Z1 > 0).float()","You can check your result yourself: build the loss from output, call loss.backward(), and compare with W1.grad"]},"cross-entropy-loss":{note:"Implement with tensor ops (torch.log, sum). Don't use F.cross_entropy, F.nll_loss or nn.CrossEntropyLoss.",starterCode:`import torch

def cross_entropy_loss(Y_pred, Y_true):
    """
    Compute cross-entropy loss.

    Args:
        Y_pred: Predicted probabilities, tensor (batch_size, n_classes)
        Y_true: One-hot encoded true labels, tensor (batch_size, n_classes)

    Returns:
        loss: Scalar cross-entropy loss as a Python float (rounded to 4 decimals)
    """
    # Your code here
    pass
`,solution:`import torch

def cross_entropy_loss(Y_pred, Y_true):
    m = Y_pred.shape[0]
    epsilon = 1e-15
    log_probs = torch.log(Y_pred + epsilon)
    loss = -(Y_true * log_probs).sum() / m
    return round(loss.item(), 4)
`,testCases:[{id:"1",description:"Perfect prediction",input:"abs(cross_entropy_loss(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]])))",expected:"0.0",hidden:!1},{id:"2",description:"Typical case",input:"cross_entropy_loss(torch.tensor([[0.7, 0.3], [0.2, 0.8]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))",expected:"0.2899",hidden:!1},{id:"3",description:"Three classes",input:"cross_entropy_loss(torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]), torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))",expected:"0.4581",hidden:!0},{id:"4",description:"Matches the negative log-likelihood of random probabilities",input:`(lambda: (
            p := torch.softmax(torch.randn(6, 4), dim=1),
            labels := torch.tensor([0, 3, 1, 2, 2, 0]),
            ref := -torch.log(p[torch.arange(6), labels]).mean().item(),
            bool(abs(cross_entropy_loss(p, torch.nn.functional.one_hot(labels, 4).float()) - ref) < 1e-3)
        )[-1])()`,expected:"True",hidden:!0}],hints:["torch.log(Y_pred + 1e-15) avoids log(0)","Multiply element-wise by Y_true, sum, negate and divide by the batch size","Return a Python float: round(loss.item(), 4)"]},"weight-init":{note:"Scale torch.randn yourself. Don't use nn.init.xavier_normal_ or nn.init.kaiming_normal_ (and note that torch reads fan_in from dim 1 of a weight).",starterCode:`import torch

def initialize_weights(n_in, n_out, method='xavier'):
    """
    Initialize weights using Xavier or He initialization.

    Args:
        n_in: Number of input units
        n_out: Number of output units
        method: 'xavier' or 'he'

    Returns:
        std: Standard deviation of initialized weights as a Python float (rounded to 4 decimals)
    """
    torch.manual_seed(42)  # For reproducibility
    # Your code here
    pass
`,solution:`import math
import torch

def initialize_weights(n_in, n_out, method='xavier'):
    torch.manual_seed(42)

    if method == 'xavier':
        std = math.sqrt(2.0 / (n_in + n_out))
    elif method == 'he':
        std = math.sqrt(2.0 / n_in)
    else:
        raise ValueError("Method must be 'xavier' or 'he'")

    W = torch.randn(n_in, n_out) * std
    return round(W.std().item(), 4)
`,testCases:[{id:"1",description:"Xavier std close to sqrt(2 / (n_in + n_out)) = 0.1155",input:'bool(abs(initialize_weights(100, 50, "xavier") - 0.1155) < 0.005)',expected:"True",hidden:!1},{id:"2",description:"He std close to sqrt(2 / n_in) = 0.1414",input:'bool(abs(initialize_weights(100, 50, "he") - 0.1414) < 0.005)',expected:"True",hidden:!1},{id:"3",description:"He on a large layer: std close to sqrt(2 / 1000) = 0.0447",input:'bool(abs(initialize_weights(1000, 500, "he") - 0.0447) < 0.0005)',expected:"True",hidden:!0},{id:"4",description:"Xavier on a large layer: std close to sqrt(2 / 1500) = 0.0365",input:'bool(abs(initialize_weights(1000, 500, "xavier") - 0.0365) < 0.0005)',expected:"True",hidden:!0}],hints:["torch.randn(n_in, n_out) samples from N(0, 1)","Xavier: multiply by sqrt(2 / (n_in + n_out)); He: multiply by sqrt(2 / n_in)","Return W.std().item() rounded to 4 decimals"]},"batch-norm":{note:"Implement with tensor ops (mean, var(unbiased=False)). Don't use F.batch_norm or nn.BatchNorm1d.",starterCode:`import torch

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    """
    Batch normalization forward pass.

    Args:
        X: Input tensor (batch_size, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant for numerical stability

    Returns:
        out: Normalized output (batch_size, features)
        cache: Tuple (X, X_norm, mu, var, gamma, eps) for backward pass
    """
    # Your code here
    pass
`,solution:`import torch

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    # Batch statistics (biased variance, like BatchNorm in training mode)
    mu = X.mean(dim=0)
    var = X.var(dim=0, unbiased=False)

    X_norm = (X - mu) / torch.sqrt(var + eps)
    out = gamma * X_norm + beta

    cache = (X, X_norm, mu, var, gamma, eps)
    return out, cache
`,testCases:[{id:"1",description:"Output mean near zero",input:"bool(torch.allclose(batch_norm_forward(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), torch.ones(2), torch.zeros(2))[0].mean(dim=0), torch.zeros(2), atol=1e-5))",expected:"True",hidden:!1},{id:"2",description:"Cache holds the batch mean and (biased) variance",input:`(lambda: (
            cache := batch_norm_forward(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), torch.ones(2), torch.zeros(2))[1],
            bool(torch.allclose(cache[2], torch.tensor([3.0, 4.0])) and torch.allclose(cache[3], torch.tensor([8.0 / 3, 8.0 / 3])))
        )[-1])()`,expected:"True",hidden:!1},{id:"3",description:"Gamma scales output",input:"list(round(v, 1) for v in batch_norm_forward(torch.tensor([[0.0, 0.0], [2.0, 2.0]]), torch.tensor([2.0, 3.0]), torch.zeros(2))[0][1].tolist())",expected:"[2.0, 3.0]",hidden:!0},{id:"4",description:"Matches F.batch_norm (training mode) on random data",input:`(lambda: (
            X := torch.randn(8, 5) * 3 + 1,
            g := torch.rand(5) + 0.5,
            b := torch.randn(5),
            bool(torch.allclose(batch_norm_forward(X, g, b)[0], torch.nn.functional.batch_norm(X, None, None, g, b, training=True, eps=1e-5), atol=1e-4))
        )[-1])()`,expected:"True",hidden:!0}],hints:["Use X.mean(dim=0) and X.var(dim=0, unbiased=False): BatchNorm uses the biased variance","Normalize: (X - mu) / torch.sqrt(var + eps)","Scale and shift: gamma * X_norm + beta"]},dropout:{note:"Build the mask yourself with torch.rand_like. Don't use F.dropout or nn.Dropout.",starterCode:`import torch

def dropout_forward(X, keep_prob=0.5, training=True):
    """
    Apply (inverted) dropout to input.

    Args:
        X: Input tensor
        keep_prob: Probability of keeping a neuron
        training: Whether in training mode

    Returns:
        out: Output after dropout
        mask: Dropout mask (for backward pass), None at inference
    """
    # Your code here
    pass
`,solution:`import torch

def dropout_forward(X, keep_prob=0.5, training=True):
    if not training:
        return X, None

    # Keep each unit with probability keep_prob, scale survivors by 1/keep_prob
    mask = (torch.rand_like(X) < keep_prob).float() / keep_prob
    out = X * mask
    return out, mask
`,testCases:[{id:"1",description:"Inference returns unchanged",input:"dropout_forward(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), 0.5, False)[0].tolist()",expected:"[[1.0, 2.0, 3.0, 4.0]]",hidden:!1},{id:"2",description:"About keep_prob of the units survive",input:"bool(abs((dropout_forward(torch.ones(1000, 100), 0.8, True)[0] != 0).float().mean().item() - 0.8) < 0.02)",expected:"True",hidden:!1},{id:"3",description:"Expected value preserved",input:"bool(abs(dropout_forward(torch.ones(1000, 100), 0.8, True)[0].mean().item() - 1.0) < 0.05)",expected:"True",hidden:!0},{id:"4",description:"Survivors are scaled by 1/keep_prob and out = X * mask",input:`(lambda: (
            X := torch.randn(50, 20),
            r := dropout_forward(X, 0.5, True),
            bool(torch.all(torch.logical_or(r[1] == 0, r[1] == 2.0)) and torch.allclose(r[0], X * r[1]))
        )[-1])()`,expected:"True",hidden:!0}],hints:["torch.rand_like(X) < keep_prob gives a boolean keep-mask","Convert with .float() and divide by keep_prob (inverted dropout)","At inference, return X unchanged (and None for the mask)"]}},Ux={},Hx={"conv2d-forward":{note:"Implement with slicing and tensor ops. Don't use F.conv2d, nn.Conv2d or unfold. A hidden test compares your output with F.conv2d.",starterCode:`import torch

def conv2d(image, kernel):
    """
    Apply 2D convolution (cross-correlation, no padding, stride 1) to an image.

    Args:
        image: Input image tensor (H, W)
        kernel: Convolution kernel tensor (kH, kW)

    Returns:
        output: Convolved image tensor (H-kH+1, W-kW+1)
    """
    # Your code here
    pass
`,solution:`import torch

def conv2d(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    out_H, out_W = H - kH + 1, W - kW + 1

    # Slide the image under each kernel tap and accumulate:
    # out[i, j] = sum_{a, b} image[i + a, j + b] * kernel[a, b]
    output = torch.zeros(out_H, out_W)
    for a in range(kH):
        for b in range(kW):
            output = output + kernel[a, b] * image[a:a + out_H, b:b + out_W]
    return output
`,testCases:[{id:"1",description:"Identity kernel",input:"conv2d(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])).tolist()",expected:"[[5.0]]",hidden:!1},{id:"2",description:"Edge detection",input:"conv2d(torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]), torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])).tolist()",expected:"[[5.0, 5.0], [5.0, 5.0]]",hidden:!1},{id:"3",description:"Non-square kernel output shape",input:"tuple(conv2d(torch.randn(6, 5), torch.randn(2, 3)).shape)",expected:"(5, 3)",hidden:!0},{id:"4",description:"Matches F.conv2d on random input",input:`(lambda: (
            img := torch.randn(7, 6),
            k := torch.randn(3, 2),
            bool(torch.allclose(conv2d(img, k), torch.nn.functional.conv2d(img[None, None], k[None, None])[0, 0], atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0}],hints:["Output size is (H-kH+1, W-kW+1)","Nested loops over output positions work: (image[i:i+kH, j:j+kW] * kernel).sum()","Faster: loop over the kH*kW kernel taps and add kernel[a, b] * image[a:a+out_H, b:b+out_W]"]},"max-pool":{note:"Implement with reshape/slicing and max. Don't use F.max_pool2d or nn.MaxPool2d.",starterCode:`import torch

def max_pool2d(image, pool_size=2):
    """
    Apply 2D max pooling (window = stride = pool_size).

    Args:
        image: Input image tensor (H, W)
        pool_size: Size of pooling window

    Returns:
        output: Pooled image tensor (H//pool_size, W//pool_size)
    """
    # Your code here
    pass
`,solution:`import torch

def max_pool2d(image, pool_size=2):
    H, W = image.shape
    p = pool_size
    out_H, out_W = H // p, W // p

    # Crop to a multiple of p, split each axis into (blocks, p), take the max per block
    x = image[:out_H * p, :out_W * p].reshape(out_H, p, out_W, p)
    return x.max(dim=3)[0].max(dim=1)[0]
`,testCases:[{id:"1",description:"4x4 to 2x2",input:"max_pool2d(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])).tolist()",expected:"[[6.0, 8.0], [14.0, 16.0]]",hidden:!1},{id:"2",description:"With negative values",input:"max_pool2d(-torch.arange(1, 17).float().reshape(4, 4)).tolist()",expected:"[[-1.0, -3.0], [-9.0, -11.0]]",hidden:!0},{id:"3",description:"Odd size drops the last row/column",input:"max_pool2d(torch.arange(25).float().reshape(5, 5)).tolist()",expected:"[[6.0, 8.0], [16.0, 18.0]]",hidden:!0},{id:"4",description:"Matches F.max_pool2d with pool_size=3",input:`(lambda: (
            img := torch.randn(9, 7),
            bool(torch.allclose(max_pool2d(img, 3), torch.nn.functional.max_pool2d(img[None, None], 3)[0, 0]))
        )[-1])()`,expected:"True",hidden:!0}],hints:["Output size is (H // pool_size, W // pool_size)","Loop version: output[i, j] = image[i*p:(i+1)*p, j*p:(j+1)*p].max()","Vectorized: reshape to (out_H, p, out_W, p) and take .max(dim=...) twice"]},"flatten-layer":{note:"Use view/reshape. nn.Flatten would do it for you; write the reshape yourself.",starterCode:`import torch

def flatten(X):
    """
    Flatten feature maps to vectors.

    Args:
        X: Input tensor (batch, height, width, channels)

    Returns:
        output: Flattened tensor (batch, height*width*channels)
    """
    # Your code here
    pass
`,solution:`import torch

def flatten(X):
    batch_size = X.shape[0]
    return X.reshape(batch_size, -1)
`,testCases:[{id:"1",description:"Correct output shape",input:"flatten(torch.randn(2, 4, 4, 3)).shape",expected:"(2, 48)",hidden:!1},{id:"2",description:"Row-major order is kept",input:"flatten(torch.arange(12).reshape(2, 3, 2, 1)).tolist()",expected:"[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]",hidden:!1},{id:"3",description:"Values preserved",input:"bool(torch.equal(flatten(torch.arange(24).reshape(1, 2, 3, 4)), torch.arange(24).reshape(1, 24)))",expected:"True",hidden:!0}],hints:["Batch size is X.shape[0]","X.reshape(batch_size, -1) infers the remaining size","X.view(X.size(0), -1) also works on contiguous tensors"]},"conv-output-size":{note:"Plain integer arithmetic. A hidden test compares your formula with the shape F.conv2d actually produces.",starterCode:`import torch

def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    """
    Calculate output size of a convolution layer.

    Args:
        input_size: Input dimension (height or width)
        kernel_size: Kernel dimension
        padding: Zero-padding
        stride: Stride

    Returns:
        output_size: Output dimension (int)
    """
    # Your code here
    pass
`,solution:`import torch

def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    return (input_size - kernel_size + 2 * padding) // stride + 1
`,testCases:[{id:"1",description:"Same padding",input:"conv_output_size(32, 3, 1, 1)",expected:"32",hidden:!1},{id:"2",description:"No padding, stride 2",input:"conv_output_size(32, 3, 0, 2)",expected:"15",hidden:!1},{id:"3",description:"Large kernel",input:"conv_output_size(224, 7, 3, 2)",expected:"112",hidden:!0},{id:"4",description:"Agrees with the shape F.conv2d produces",input:"conv_output_size(11, 4, 2, 3) == torch.nn.functional.conv2d(torch.zeros(1, 1, 11, 11), torch.zeros(1, 1, 4, 4), padding=2, stride=3).shape[-1]",expected:"True",hidden:!0}],hints:["Apply the formula: (input - kernel + 2*padding) / stride + 1","Use integer division (//) for the result"]},"conv2d-advanced":{note:"Implement with tensor ops (F.pad for the padding is fine). Don't use F.conv2d, nn.Conv2d or unfold. Hidden tests compare against F.conv2d.",starterCode:`import torch
import torch.nn.functional as F

def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    """
    Advanced 2D convolution with padding, stride, and groups.

    Args:
        input: Input tensor (batch, in_channels, H, W)
        kernel: Convolution kernels (out_channels, in_channels//groups, kH, kW)
        padding: Zero-padding added to input (default: 0)
        stride: Stride of the convolution (default: 1)
        groups: Number of groups for grouped convolution (default: 1)

    Returns:
        output: Convolved tensor (batch, out_channels, H_out, W_out)
    """
    # Your code here
    pass
`,solution:`import torch
import torch.nn.functional as F

def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))

    B, C_in, H, W = input.shape
    C_out, C_g, kH, kW = kernel.shape
    G = groups
    O_g = C_out // G
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    # Split channels into groups: input (B, G, C_g, H, W), kernel (G, O_g, C_g, kH, kW)
    x = input.reshape(B, G, C_g, H, W)
    k = kernel.reshape(G, O_g, C_g, kH, kW)

    out = torch.zeros(B, G, O_g, H_out, W_out)
    for a in range(kH):
        for b in range(kW):
            # Input pixels seen by kernel tap (a, b) at every output position
            patch = x[:, :, :, a:a + stride * (H_out - 1) + 1:stride, b:b + stride * (W_out - 1) + 1:stride]
            out = out + torch.einsum('bgchw,goc->bgohw', patch, k[:, :, :, a, b])

    return out.reshape(B, C_out, H_out, W_out)
`,testCases:[{id:"1",description:"Basic convolution with padding and stride",input:"conv2d_advanced(torch.ones(1, 1, 5, 5), torch.ones(1, 1, 3, 3), 1, 2, 1).shape",expected:"(1, 1, 3, 3)",hidden:!1},{id:"2",description:"Padding and stride values (hand-computed)",input:"conv2d_advanced(torch.ones(1, 1, 3, 3), torch.ones(1, 1, 2, 2), 1, 2, 1).tolist()",expected:"[[[[1.0, 2.0], [2.0, 4.0]]]]",hidden:!1},{id:"3",description:"Grouped convolution (groups=2) shape",input:"conv2d_advanced(torch.arange(32).reshape(1, 2, 4, 4).float(), torch.ones(4, 1, 2, 2), 0, 1, 2).shape",expected:"(1, 4, 3, 3)",hidden:!1},{id:"4",description:"Depthwise convolution (groups=in_channels)",input:"conv2d_advanced(torch.ones(2, 3, 4, 4), torch.ones(3, 1, 2, 2), 0, 1, 3).shape",expected:"(2, 3, 3, 3)",hidden:!0},{id:"5",description:"Verify correct output values with groups",input:"bool(torch.allclose(conv2d_advanced(torch.ones(1, 2, 3, 3), torch.ones(2, 1, 2, 2), 0, 1, 2), torch.full((1, 2, 2, 2), 4.0)))",expected:"True",hidden:!0},{id:"6",description:"Matches F.conv2d (padding=1, stride=2, groups=2) on random input",input:`(lambda: (
            x := torch.randn(2, 4, 7, 6),
            k := torch.randn(6, 2, 3, 2),
            bool(torch.allclose(conv2d_advanced(x, k, 1, 2, 2), torch.nn.functional.conv2d(x, k, padding=1, stride=2, groups=2), atol=1e-4))
        )[-1])()`,expected:"True",hidden:!0}],hints:["Pad with F.pad(input, (p, p, p, p)): it pads the last two dims","H_out = (H_padded - kH) // stride + 1","For groups: reshape input to (B, G, C_in//G, H, W) and kernel to (G, C_out//G, C_in//G, kH, kW)","Loop over kernel taps (a, b): take the strided slice x[..., a::stride, b::stride] (cropped to H_out, W_out) and contract channels with einsum"]}},Qx={},Xx={"scaled-dot-product-attention":{note:"Implement with matmul and softmax. Don't use F.scaled_dot_product_attention or nn.MultiheadAttention; a hidden test compares against F.scaled_dot_product_attention.",starterCode:`import math
import torch

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Queries (seq_len, d_k)
        K: Keys (seq_len, d_k)
        V: Values (seq_len, d_v)

    Returns:
        output: Attention output (seq_len, d_v)
        attention_weights: Attention weights (seq_len, seq_len)
    """
    # Your code here
    pass
`,solution:`import math
import torch

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]

    # Similarity of every query with every key, scaled by sqrt(d_k)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    # Each row becomes a probability distribution over keys
    attention_weights = torch.softmax(scores, dim=-1)

    output = attention_weights @ V
    return output, attention_weights
`,testCases:[{id:"1",description:"Output shape correct",input:"scaled_dot_product_attention(torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8))[0].shape",expected:"(4, 8)",hidden:!1},{id:"2",description:"Attention weights sum to 1",input:"bool(torch.allclose(scaled_dot_product_attention(torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8))[1].sum(dim=-1), torch.ones(4)))",expected:"True",hidden:!1},{id:"3",description:"Zero queries attend uniformly: output is the mean of V",input:`(lambda: (
            V := torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            bool(torch.allclose(scaled_dot_product_attention(torch.zeros(3, 4), torch.randn(3, 4), V)[0], torch.tensor([[3.0, 4.0]] * 3), atol=1e-5))
        )[-1])()`,expected:"True",hidden:!1},{id:"4",description:"Matches F.scaled_dot_product_attention",input:`(lambda: (
            Q := torch.randn(5, 8), K := torch.randn(5, 8), V := torch.randn(5, 3),
            bool(torch.allclose(scaled_dot_product_attention(Q, K, V)[0], torch.nn.functional.scaled_dot_product_attention(Q[None], K[None], V[None])[0], atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0},{id:"5",description:"Scaling by sqrt(d_k) is applied",input:`(lambda: (
            Q := torch.tensor([[2.0, 0.0, 0.0, 0.0]]), K := torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            round(scaled_dot_product_attention(Q, K, torch.randn(2, 2))[1][0, 0].item(), 4)
        )[-1])()`,expected:"0.7311",hidden:!0}],hints:["scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)","torch.softmax(scores, dim=-1) normalizes each query row","Return (weights @ V, weights)"]},"multi-head-attention":{note:"Split heads with view/transpose and implement attention yourself. Don't use nn.MultiheadAttention or F.scaled_dot_product_attention.",starterCode:`import math
import torch

def multi_head_attention(Q, K, V, num_heads):
    """
    Multi-head attention mechanism (no learned projections).

    Args:
        Q: Queries (seq_len, d_model)
        K: Keys (seq_len, d_model)
        V: Values (seq_len, d_model)
        num_heads: Number of attention heads

    Returns:
        output: Multi-head attention output (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    # Your code here
    pass
`,solution:`import math
import torch

def multi_head_attention(Q, K, V, num_heads):
    seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # (seq_len, d_model) -> (num_heads, seq_len, d_k)
    def split_heads(x):
        return x.reshape(seq_len, num_heads, d_k).transpose(0, 1)

    Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

    # Attention for all heads at once (batched matmul over the head dim)
    scores = Qh @ Kh.transpose(-2, -1) / math.sqrt(d_k)
    attention = torch.softmax(scores, dim=-1)
    heads = attention @ Vh  # (num_heads, seq_len, d_k)

    # Concatenate heads: (num_heads, seq_len, d_k) -> (seq_len, d_model)
    return heads.transpose(0, 1).reshape(seq_len, d_model)
`,testCases:[{id:"1",description:"Output shape preserved",input:"multi_head_attention(torch.randn(4, 64), torch.randn(4, 64), torch.randn(4, 64), 8).shape",expected:"(4, 64)",hidden:!1},{id:"2",description:"Zero queries attend uniformly: output rows equal the mean of V",input:`(lambda: (
            V := torch.randn(3, 4),
            bool(torch.allclose(multi_head_attention(torch.zeros(3, 4), torch.randn(3, 4), V, 2), V.mean(dim=0).expand(3, 4), atol=1e-5))
        )[-1])()`,expected:"True",hidden:!1},{id:"3",description:"Matches running attention separately on each head and concatenating",input:`(lambda: (
            Q := torch.randn(5, 16), K := torch.randn(5, 16), V := torch.randn(5, 16),
            ref := torch.cat([torch.softmax(Q[:, h * 4:(h + 1) * 4] @ K[:, h * 4:(h + 1) * 4].T / 2.0, dim=-1) @ V[:, h * 4:(h + 1) * 4] for h in range(4)], dim=1),
            bool(torch.allclose(multi_head_attention(Q, K, V, 4), ref, atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0},{id:"4",description:"One head equals plain scaled dot-product attention",input:`(lambda: (
            Q := torch.randn(4, 8), K := torch.randn(4, 8), V := torch.randn(4, 8),
            bool(torch.allclose(multi_head_attention(Q, K, V, 1), torch.softmax(Q @ K.T / 8 ** 0.5, dim=-1) @ V, atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0}],hints:["Reshape to (seq_len, num_heads, d_k), then transpose(0, 1) to (num_heads, seq_len, d_k)","Batched matmul handles all heads at once: Qh @ Kh.transpose(-2, -1)","Transpose back and reshape to (seq_len, d_model)"]},"positional-encoding":{note:"Build the table with torch.arange, torch.exp, torch.sin/cos and strided slice assignment (PE[:, 0::2] = ...).",starterCode:`import math
import torch

def positional_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encodings.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        PE: Positional encoding tensor (max_len, d_model)
    """
    # Your code here
    pass
`,solution:`import math
import torch

def positional_encoding(max_len, d_model):
    PE = torch.zeros(max_len, d_model)

    position = torch.arange(max_len).float().unsqueeze(1)  # (max_len, 1)
    # 1 / 10000^(2i/d_model), computed in log space
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)
    return PE
`,testCases:[{id:"1",description:"Correct shape",input:"positional_encoding(50, 64).shape",expected:"(50, 64)",hidden:!1},{id:"2",description:"Position 1 = [sin(1), cos(1), sin(0.01), cos(0.01)] for d_model=4",input:"bool(torch.allclose(positional_encoding(10, 4)[1], torch.tensor([0.841471, 0.540302, 0.0099998, 0.99995]), atol=1e-5))",expected:"True",hidden:!1},{id:"3",description:"Position 0 alternates sin(0)=0 and cos(0)=1",input:"positional_encoding(10, 6)[0].tolist()",expected:"[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]",hidden:!0},{id:"4",description:"Values for a larger table",input:`(lambda: (
            PE := positional_encoding(20, 8),
            bool(abs(PE[7, 2].item() - 0.6442177) < 1e-5 and abs(PE[19, 5].item() - 0.9820042) < 1e-5)
        )[-1])()`,expected:"True",hidden:!0}],hints:["position = torch.arange(max_len).float().unsqueeze(1)","div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))","PE[:, 0::2] = torch.sin(position * div_term); PE[:, 1::2] = torch.cos(position * div_term)"]},"layer-norm":{note:"Implement with mean and var(unbiased=False) over the last dim. Don't use F.layer_norm or nn.LayerNorm; a hidden test compares against F.layer_norm.",starterCode:`import torch

def layer_norm(X, gamma, beta, eps=1e-5):
    """
    Apply layer normalization over the last dimension.

    Args:
        X: Input (batch, seq_len, features) or (seq_len, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant

    Returns:
        output: Normalized tensor (same shape as X)
    """
    # Your code here
    pass
`,solution:`import torch

def layer_norm(X, gamma, beta, eps=1e-5):
    # Statistics over the feature (last) dimension, per position
    mean = X.mean(dim=-1, keepdim=True)
    var = X.var(dim=-1, keepdim=True, unbiased=False)

    X_norm = (X - mean) / torch.sqrt(var + eps)
    return gamma * X_norm + beta
`,testCases:[{id:"1",description:"Output mean near zero",input:"bool(torch.allclose(layer_norm(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), torch.ones(4), torch.zeros(4)).mean(dim=-1), torch.zeros(2), atol=1e-5))",expected:"True",hidden:!1},{id:"2",description:"Gamma and beta work",input:"bool(torch.allclose(layer_norm(torch.tensor([[0.0, 0.0, 2.0, 2.0]]), torch.tensor([1.0, 2.0, 1.0, 2.0]), torch.tensor([0.5, 0.0, 0.0, 0.0])), torch.tensor([[-0.5, -2.0, 1.0, 2.0]]), atol=1e-4))",expected:"True",hidden:!1},{id:"3",description:"Uses the biased variance (unit variance output)",input:"round(layer_norm(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.ones(4), torch.zeros(4)).var(unbiased=False).item(), 3)",expected:"1.0",hidden:!0},{id:"4",description:"Matches F.layer_norm on a 3-D input",input:`(lambda: (
            X := torch.randn(2, 5, 8) * 2 + 1, g := torch.randn(8), b := torch.randn(8),
            bool(torch.allclose(layer_norm(X, g, b), torch.nn.functional.layer_norm(X, (8,), g, b, eps=1e-5), atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0}],hints:["X.mean(dim=-1, keepdim=True) and X.var(dim=-1, keepdim=True, unbiased=False)","keepdim=True keeps the shapes broadcastable","Normalize, then gamma * X_norm + beta"]},"causal-mask":{note:"Use torch.triu / torch.full (or masked_fill). The mask is additive: 0 where attention is allowed, -inf above the diagonal.",starterCode:`import torch

def create_causal_mask(seq_len):
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length

    Returns:
        mask: Causal mask tensor (seq_len, seq_len)
              0 for allowed positions, -inf for masked
    """
    # Your code here
    pass
`,solution:`import torch

def create_causal_mask(seq_len):
    # -inf strictly above the diagonal, 0 on and below it
    return torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
`,testCases:[{id:"1",description:"Correct shape",input:"create_causal_mask(4).shape",expected:"(4, 4)",hidden:!1},{id:"2",description:"Upper triangle is -inf",input:"torch.isinf(create_causal_mask(3)).tolist()",expected:"[[false, true, true], [false, false, true], [false, false, false]]",hidden:!1},{id:"3",description:"Lower triangle (incl. diagonal) is 0, upper is -inf",input:'bool(torch.allclose(create_causal_mask(3), torch.tensor([[0.0, float("-inf"), float("-inf")], [0.0, 0.0, float("-inf")], [0.0, 0.0, 0.0]])))',expected:"True",hidden:!0},{id:"4",description:"After softmax, row 1 attends equally to positions 0 and 1 only",input:"torch.softmax(create_causal_mask(4), dim=-1)[1].tolist()",expected:"[0.5, 0.5, 0.0, 0.0]",hidden:!0}],hints:["torch.full((n, n), float('-inf')) fills a matrix with -inf","torch.triu(x, diagonal=1) keeps only the part strictly above the diagonal (the rest becomes 0)",'Alternative: torch.zeros(n, n).masked_fill(torch.ones(n, n).triu(1).bool(), float("-inf"))']}},Kx={},Yx={},$x={"vae-reparameterization":{note:"Sample eps with torch.randn_like. Don't use torch.distributions.Normal(...).rsample(), which does the trick for you. Hidden tests check that gradients flow to mu and log_var.",starterCode:`import torch

def reparameterize(mu, log_var):
    """
    Sample from latent distribution using reparameterization trick.

    Args:
        mu: Mean of latent distribution (batch, latent_dim)
        log_var: Log variance of latent distribution (batch, latent_dim)

    Returns:
        z: Sampled latent vectors (batch, latent_dim), differentiable w.r.t. mu and log_var
    """
    torch.manual_seed(42)  # For reproducibility
    # Your code here
    pass
`,solution:`import torch

def reparameterize(mu, log_var):
    torch.manual_seed(42)
    std = torch.exp(0.5 * log_var)
    # The randomness lives in eps, which does not depend on mu or log_var
    eps = torch.randn_like(std)
    return mu + std * eps
`,testCases:[{id:"1",description:"Output shape matches input",input:"reparameterize(torch.zeros(2, 4), torch.zeros(2, 4)).shape",expected:"(2, 4)",hidden:!1},{id:"2",description:"Samples have mean mu and std exp(0.5 * log_var)",input:`(lambda: (
            z := reparameterize(torch.full((20000, 1), 2.0), torch.full((20000, 1), 1.3862944)),
            bool(abs(z.mean().item() - 2.0) < 0.1 and abs(z.std().item() - 2.0) < 0.1)
        )[-1])()`,expected:"True",hidden:!1},{id:"3",description:"Zero variance returns mu exactly",input:"bool(torch.allclose(reparameterize(torch.ones(3, 5), torch.full((3, 5), -1000.0)), torch.ones(3, 5)))",expected:"True",hidden:!0},{id:"4",description:"Gradients flow: dz/dmu = 1 and dz/dlog_var = 0.5 * std * eps",input:`(lambda: (
            mu := torch.zeros(2, 3, requires_grad=True),
            lv := torch.tensor([[0.0, 1.0, -1.0], [0.5, 0.0, 2.0]], requires_grad=True),
            z := reparameterize(mu, lv),
            z.sum().backward(),
            bool(torch.allclose(mu.grad, torch.ones(2, 3)) and torch.allclose(lv.grad, 0.5 * (z - mu).detach(), atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0}],hints:["std = torch.exp(0.5 * log_var)","eps = torch.randn_like(std) is constant w.r.t. mu and log_var","z = mu + std * eps lets autograd differentiate through the sample"]},"vae-loss":{note:"F.mse_loss is fine for the reconstruction term; write the KL term yourself with tensor ops.",starterCode:`import torch
import torch.nn.functional as F

def vae_loss(x, x_reconstructed, mu, log_var):
    """
    Compute VAE loss (negative ELBO).

    Args:
        x: Original input (batch, features)
        x_reconstructed: Reconstructed input (batch, features)
        mu: Latent mean (batch, latent_dim)
        log_var: Latent log variance (batch, latent_dim)

    Returns:
        (total_loss, recon_loss, kl_loss) as Python floats rounded to 4 decimals
    """
    # Your code here
    pass
`,solution:`import torch
import torch.nn.functional as F

def vae_loss(x, x_reconstructed, mu, log_var):
    # Reconstruction loss (MSE, mean over all elements)
    recon_loss = F.mse_loss(x_reconstructed, x)

    # KL(q(z|x) || N(0, I)): sum over latent dims, mean over batch
    kl_loss = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)).mean()

    total_loss = recon_loss + kl_loss
    return round(total_loss.item(), 4), round(recon_loss.item(), 4), round(kl_loss.item(), 4)
`,testCases:[{id:"1",description:"Perfect case",input:"vae_loss(torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, 2), torch.zeros(2, 2))",expected:"(0.0, 0.0, 0.0)",hidden:!1},{id:"2",description:"Non-zero reconstruction and KL",input:"vae_loss(torch.tensor([[1.0, 2.0]]), torch.zeros(1, 2), torch.tensor([[0.5, -0.5]]), torch.tensor([[0.0, 1.0]]))",expected:"(3.1091, 2.5, 0.6091)",hidden:!1},{id:"3",description:"Non-zero KL",input:"vae_loss(torch.zeros(1, 4), torch.zeros(1, 4), torch.ones(1, 2), torch.zeros(1, 2))",expected:"(1.0, 0.0, 1.0)",hidden:!0},{id:"4",description:"KL sums over latent dims and averages over the batch",input:"vae_loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([[1.0, 1.0], [0.0, 0.0]]), torch.zeros(2, 2))",expected:"(0.5, 0.0, 0.5)",hidden:!0}],hints:["Reconstruction: F.mse_loss(x_reconstructed, x) or ((x - x_reconstructed) ** 2).mean()","KL per sample: -0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1)","Average the KL over the batch, then return Python floats with round(t.item(), 4)"]},"diffusion-noise-schedule":{note:"Use torch.linspace and torch.cumprod.",starterCode:`import torch

def linear_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    """
    Create linear noise schedule for diffusion.

    Args:
        T: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        betas: Beta values (T,)
        alphas: Alpha values (T,)
        alpha_bars: Cumulative alpha product (T,)
    """
    # Your code here
    pass
`,solution:`import torch

def linear_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars
`,testCases:[{id:"1",description:"Beta range correct",input:"linear_noise_schedule(100, 0.0001, 0.02)[0].shape",expected:"(100,)",hidden:!1},{id:"2",description:"Small schedule values (hand-computed)",input:`(lambda: (
            s := linear_noise_schedule(3, 0.1, 0.3),
            bool(torch.allclose(s[0], torch.tensor([0.1, 0.2, 0.3])) and torch.allclose(s[1], torch.tensor([0.9, 0.8, 0.7])) and torch.allclose(s[2], torch.tensor([0.9, 0.72, 0.504])))
        )[-1])()`,expected:"True",hidden:!1},{id:"3",description:"Alpha bar decreases - last < first",input:"bool(linear_noise_schedule(50, 0.001, 0.01)[2][-1] < linear_noise_schedule(50, 0.001, 0.01)[2][0])",expected:"True",hidden:!0},{id:"4",description:"Standard DDPM schedule endpoints",input:`(lambda: (
            s := linear_noise_schedule(1000),
            bool(abs(s[0][0].item() - 0.0001) < 1e-7 and abs(s[0][-1].item() - 0.02) < 1e-7 and abs(s[2][-1].item() - 4.036e-05) < 1e-6)
        )[-1])()`,expected:"True",hidden:!0}],hints:["torch.linspace(beta_start, beta_end, T) gives T evenly spaced betas","alphas = 1 - betas","torch.cumprod(alphas, dim=0) gives the running product"]},"diffusion-forward":{note:"Sample the noise with torch.randn_like. Tests check the formula exactly, so return the same noise you used.",starterCode:`import torch

def diffusion_forward(x_0, t, alpha_bars):
    """
    Add noise to data using forward diffusion.

    Args:
        x_0: Original data (batch, ...)
        t: Timestep (int)
        alpha_bars: Cumulative alpha products (T,)

    Returns:
        x_t: Noisy data at timestep t
        noise: The noise that was added
    """
    torch.manual_seed(42)
    # Your code here
    pass
`,solution:`import torch

def diffusion_forward(x_0, t, alpha_bars):
    torch.manual_seed(42)

    alpha_bar_t = alpha_bars[t]
    noise = torch.randn_like(x_0)

    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    return x_t, noise
`,testCases:[{id:"1",description:"Output shape matches input",input:"diffusion_forward(torch.ones(2, 4), 50, torch.linspace(0.99, 0.01, 100))[0].shape",expected:"(2, 4)",hidden:!1},{id:"2",description:"x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise",input:`(lambda: (
            x0 := torch.randn(2, 4), ab := torch.linspace(0.99, 0.01, 100),
            r := diffusion_forward(x0, 50, ab),
            bool(torch.allclose(r[0], ab[50].sqrt() * x0 + (1 - ab[50]).sqrt() * r[1], atol=1e-6))
        )[-1])()`,expected:"True",hidden:!1},{id:"3",description:"Noise shape matches input",input:"diffusion_forward(torch.ones(3, 5), 10, torch.linspace(0.99, 0.01, 100))[1].shape",expected:"(3, 5)",hidden:!0},{id:"4",description:"alpha_bar = 1 keeps x_0; alpha_bar = 0 gives pure noise",input:`(lambda: (
            x0 := torch.randn(4, 3), ab := torch.tensor([1.0, 0.0]),
            r0 := diffusion_forward(x0, 0, ab), r1 := diffusion_forward(x0, 1, ab),
            bool(torch.allclose(r0[0], x0) and torch.allclose(r1[0], r1[1]))
        )[-1])()`,expected:"True",hidden:!0},{id:"5",description:"Noise is standard normal",input:`(lambda: (
            n := diffusion_forward(torch.zeros(100, 100), 0, torch.tensor([0.5]))[1],
            bool(abs(n.mean().item()) < 0.05 and abs(n.std().item() - 1.0) < 0.05)
        )[-1])()`,expected:"True",hidden:!0}],hints:["alpha_bar_t = alpha_bars[t]","noise = torch.randn_like(x_0)","x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise"]},"vqvae-quantization":{note:"Compute distances with tensor ops and pick codes with argmin. Don't use torch.cdist. For the losses, use .detach() as the stop-gradient.",starterCode:`import torch
import torch.nn.functional as F

def vq_quantize(z_e, codebook):
    """
    Vector quantization for VQ-VAE.

    Args:
        z_e: Encoder output (batch, H, W, embedding_dim)
        codebook: Embedding vectors (num_embeddings, embedding_dim)

    Returns:
        z_q: Quantized vectors (batch, H, W, embedding_dim)
        indices: Codebook indices used (batch, H, W), int64
    """
    # Your code here
    pass


def vq_loss(z_e, z_q, beta=0.25):
    """
    Compute VQ-VAE losses (codebook + commitment).

    Args:
        z_e: Encoder output (batch, H, W, embedding_dim)
        z_q: Quantized vectors (batch, H, W, embedding_dim)
        beta: Commitment loss weight (default: 0.25)

    Returns:
        codebook_loss: Loss to update codebook (Python float, rounded to 4 decimals)
        commitment_loss: Loss to commit encoder to codebook (Python float, rounded to 4 decimals)
    """
    # Your code here
    pass
`,solution:`import torch
import torch.nn.functional as F

def vq_quantize(z_e, codebook):
    batch, H, W, D = z_e.shape

    # Flatten spatial dimensions: (N, D)
    z_flat = z_e.reshape(-1, D)

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, for every (vector, code) pair: (N, K)
    distances = (z_flat ** 2).sum(dim=1, keepdim=True) + (codebook ** 2).sum(dim=1) - 2 * z_flat @ codebook.T

    indices_flat = distances.argmin(dim=1)
    z_q = codebook[indices_flat].reshape(batch, H, W, D)
    return z_q, indices_flat.reshape(batch, H, W)


def vq_loss(z_e, z_q, beta=0.25):
    # Codebook loss: ||sg[z_e] - z_q||^2 moves the codes toward the encoder output
    codebook_loss = F.mse_loss(z_q, z_e.detach())
    # Commitment loss: ||z_e - sg[z_q]||^2 keeps the encoder close to its codes
    commitment_loss = beta * F.mse_loss(z_e, z_q.detach())
    return round(codebook_loss.item(), 4), round(commitment_loss.item(), 4)
`,testCases:[{id:"1",description:"z_q output shape matches input",input:"vq_quantize(torch.randn(2, 4, 4, 8), torch.randn(16, 8))[0].shape",expected:"(2, 4, 4, 8)",hidden:!1},{id:"2",description:"Nearest codes picked (hand-computed)",input:"vq_quantize(torch.tensor([[[[1.0, 1.0], [9.0, 8.0], [4.0, 6.0]]]]), torch.tensor([[0.0, 0.0], [10.0, 10.0], [5.0, 5.0]]))[1].tolist()",expected:"[[[0, 1, 2]]]",hidden:!1},{id:"3",description:"VQ losses (hand-computed)",input:"vq_loss(torch.tensor([[1.0, 1.0], [9.0, 8.0]]), torch.tensor([[0.0, 0.0], [10.0, 10.0]]), 0.25)",expected:"(1.75, 0.4375)",hidden:!1},{id:"4",description:"Indices match a brute-force nearest-neighbour search",input:`(lambda: (
            z := torch.randn(2, 3, 3, 4), cb := torch.randn(8, 4),
            bool(torch.equal(vq_quantize(z, cb)[1], ((z.unsqueeze(-2) - cb) ** 2).sum(dim=-1).argmin(dim=-1)))
        )[-1])()`,expected:"True",hidden:!0},{id:"5",description:"Quantized vectors are rows of the codebook",input:`(lambda: (
            z := torch.randn(2, 4, 4, 8), cb := torch.randn(16, 8),
            q := vq_quantize(z, cb),
            bool(torch.allclose(q[0], cb[q[1]]))
        )[-1])()`,expected:"True",hidden:!0}],hints:["Reshape z_e to (batch*H*W, D) to compute all distances at once","Distances: (z**2).sum(1, keepdim=True) + (codebook**2).sum(1) - 2 * z @ codebook.T","distances.argmin(dim=1) gives indices; codebook[indices] gives z_q","Losses: F.mse_loss(z_q, z_e.detach()) and beta * F.mse_loss(z_e, z_q.detach())"]},"kl-divergence":{note:"Inputs are scalar tensors. Write the closed form with torch.log; return a Python float.",starterCode:`import torch

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute KL divergence between two Gaussians.

    Args:
        mu_p, sigma_p: Mean and std of P (scalar tensors)
        mu_q, sigma_q: Mean and std of Q (scalar tensors)

    Returns:
        kl: KL(P || Q) as a Python float rounded to 4 decimals
    """
    # Your code here
    pass
`,solution:`import torch

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    term1 = torch.log(sigma_q / sigma_p)
    term2 = (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
    kl = term1 + term2 - 0.5
    return round(kl.item(), 4)
`,testCases:[{id:"1",description:"Same distribution",input:"kl_divergence_gaussian(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0))",expected:"0.0",hidden:!1},{id:"2",description:"Different means",input:"kl_divergence_gaussian(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0))",expected:"0.5",hidden:!1},{id:"3",description:"Different variances",input:"kl_divergence_gaussian(torch.tensor(0.0), torch.tensor(2.0), torch.tensor(0.0), torch.tensor(1.0))",expected:"0.8069",hidden:!0},{id:"4",description:"Not symmetric: KL(Q || P) for the previous pair",input:"kl_divergence_gaussian(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(2.0))",expected:"0.3181",hidden:!0}],hints:["KL = log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5","torch.log works on scalar tensors","Return round(kl.item(), 4)"]}},Gx={"rl-discounted-return":{note:"Vectorize with torch.arange: no Python loop needed. Inputs are tensors; return a Python float.",starterCode:`import torch

def discounted_return(rewards: torch.Tensor, gamma: float) -> float:
    """
    Calculate the discounted cumulative return.

    Args:
        rewards: Tensor of rewards [r_0, r_1, ..., r_T]
        gamma: Discount factor (0 <= gamma <= 1)

    Returns:
        Total discounted return G_0 as a Python float (rounded to 3 decimals)
    """
    rewards = rewards.double()
    # Your code here
    pass
`,solution:`import torch

def discounted_return(rewards: torch.Tensor, gamma: float) -> float:
    rewards = rewards.double()
    T = rewards.shape[0]

    # Discount factors: [gamma^0, gamma^1, gamma^2, ...]
    discounts = gamma ** torch.arange(T, dtype=torch.float64)

    G = (rewards * discounts).sum()
    return round(G.item(), 3)
`,testCases:[{id:"1",description:"Constant rewards",input:"discounted_return(torch.tensor([1.0, 1.0, 1.0, 1.0]), 0.9)",expected:"3.439",hidden:!1},{id:"2",description:"Delayed reward",input:"discounted_return(torch.tensor([0.0, 0.0, 0.0, 10.0]), 0.5)",expected:"1.25",hidden:!1},{id:"3",description:"No discount",input:"discounted_return(torch.tensor([1.0, 2.0, 3.0, 4.0]), 1.0)",expected:"10.0",hidden:!1},{id:"4",description:"Only immediate",input:"discounted_return(torch.tensor([5.0, 10.0, 15.0]), 0.0)",expected:"5.0",hidden:!0},{id:"5",description:"Mixed-sign rewards",input:"discounted_return(torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0]), 0.8)",expected:"1.32",hidden:!0}],hints:["gamma ** torch.arange(T, dtype=torch.float64) gives [γ⁰, γ¹, γ², ...]","Multiply element-wise with rewards and .sum()","Return round(G.item(), 3)"]},"rl-epsilon-greedy":{note:"Use torch.argmax for the greedy action and return a Python int.",starterCode:`import torch

def epsilon_greedy(q_values: torch.Tensor, epsilon: float,
                   random_value: float) -> int:
    """
    Select action using epsilon-greedy policy.

    Args:
        q_values: Q-values for each action [Q(a_0), Q(a_1), ...] (tensor)
        epsilon: Exploration probability (0 <= epsilon <= 1)
        random_value: Pre-generated random number for deterministic testing

    Returns:
        Selected action index (Python int)
    """
    n_actions = q_values.shape[0]

    # Your code here
    # If random_value < epsilon: explore (return floor(random_value * n_actions))
    # Else: exploit (return argmax of q_values)
    pass
`,solution:`import torch

def epsilon_greedy(q_values: torch.Tensor, epsilon: float,
                   random_value: float) -> int:
    n_actions = q_values.shape[0]

    if random_value < epsilon:
        # Explore: random action
        return int(random_value * n_actions)
    # Exploit: greedy action
    return int(torch.argmax(q_values).item())
`,testCases:[{id:"1",description:"Exploit (greedy)",input:"epsilon_greedy(torch.tensor([1.0, 3.0, 2.0]), 0.1, 0.5)",expected:"1",hidden:!1},{id:"2",description:"Explore",input:"epsilon_greedy(torch.tensor([1.0, 3.0, 2.0]), 0.9, 0.5)",expected:"1",hidden:!1},{id:"3",description:"Always exploit (epsilon=0)",input:"epsilon_greedy(torch.tensor([5.0, 2.0, 8.0, 1.0]), 0.0, 0.99)",expected:"2",hidden:!1},{id:"4",description:"Explore with low random",input:"epsilon_greedy(torch.tensor([1.0, 2.0, 3.0, 4.0]), 0.5, 0.2)",expected:"0",hidden:!0},{id:"5",description:"Explore picks floor(random_value * n_actions)",input:"epsilon_greedy(torch.tensor([9.0, 0.0, 0.0, 0.0, 0.0]), 1.0, 0.75)",expected:"3",hidden:!0}],hints:["Compare random_value to epsilon to decide explore vs exploit","For exploration, compute int(random_value * n_actions)","For exploitation, use int(torch.argmax(q_values))"]},"rl-bellman-value":{note:"Gather V(s') with tensor indexing and take the probability-weighted sum. Return a Python float.",starterCode:`import torch

def bellman_value(transitions: list, gamma: float,
                  next_values: torch.Tensor) -> float:
    """
    Compute state value using Bellman expectation equation.

    Args:
        transitions: List of (probability, reward, next_state_index) tuples
        gamma: Discount factor
        next_values: Tensor of values V(s') for each state

    Returns:
        Value V(s) for the current state (Python float, rounded to 2 decimals)
    """
    next_values = next_values.double()
    # Your code here
    pass
`,solution:`import torch

def bellman_value(transitions: list, gamma: float,
                  next_values: torch.Tensor) -> float:
    next_values = next_values.double()

    probs = torch.tensor([float(p) for p, _, _ in transitions], dtype=torch.float64)
    rewards = torch.tensor([float(r) for _, r, _ in transitions], dtype=torch.float64)
    next_states = torch.tensor([s for _, _, s in transitions])

    # V(s) = sum_s' P(s'|s) * [R + gamma * V(s')]
    value = (probs * (rewards + gamma * next_values[next_states])).sum()
    return round(value.item(), 2)
`,testCases:[{id:"1",description:"Two transitions",input:"bellman_value([(0.5, 1, 0), (0.5, 2, 1)], 0.9, torch.tensor([5.0, 10.0]))",expected:"8.25",hidden:!1},{id:"2",description:"Deterministic transition",input:"bellman_value([(1.0, 5, 0)], 0.9, torch.tensor([10.0]))",expected:"14.0",hidden:!1},{id:"3",description:"Three outcomes",input:"bellman_value([(0.2, 0, 0), (0.5, 1, 1), (0.3, 2, 2)], 0.9, torch.tensor([0.0, 5.0, 10.0]))",expected:"6.05",hidden:!0},{id:"4",description:"Repeated next state",input:"bellman_value([(0.25, 4, 1), (0.75, -1, 1)], 0.5, torch.tensor([100.0, 2.0]))",expected:"1.25",hidden:!0}],hints:["Build tensors of probabilities, rewards and next-state indices from the transitions","next_values[next_states] gathers V(s') for every transition","Return round((probs * (rewards + gamma * v_next)).sum().item(), 2)"]},"rl-q-learning-update":{note:"Index the Q-table tensor directly. Don't modify the input: work on Q.clone(). Return the updated Q-table tensor.",starterCode:`import torch

def q_learning_update(Q: torch.Tensor, state: int, action: int,
                      reward: float, next_state: int,
                      alpha: float, gamma: float) -> torch.Tensor:
    """
    Perform one Q-learning update.

    Args:
        Q: Q-table tensor of shape (n_states, n_actions)
        state: Current state index
        action: Action taken
        reward: Reward received
        next_state: Next state index
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        Updated Q-table (new tensor; the input is left unchanged)
    """
    Q = Q.clone().float()
    # Your code here
    pass
`,solution:`import torch

def q_learning_update(Q: torch.Tensor, state: int, action: int,
                      reward: float, next_state: int,
                      alpha: float, gamma: float) -> torch.Tensor:
    Q = Q.clone().float()

    # Bellman optimality target uses the best next action
    td_target = reward + gamma * Q[next_state].max()
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error
    return Q
`,testCases:[{id:"1",description:"Simple update",input:"bool(torch.allclose(q_learning_update(torch.zeros(2, 2), 0, 1, 1.0, 1, 0.1, 0.9), torch.tensor([[0.0, 0.1], [0.0, 0.0]])))",expected:"True",hidden:!1},{id:"2",description:"With existing values: Q[0, 0] = 1 + 0.5 * (5 + 0.9 * 4 - 1)",input:"round(q_learning_update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, 0, 5.0, 1, 0.5, 0.9)[0, 0].item(), 4)",expected:"4.8",hidden:!1},{id:"3",description:"Only Q[state, action] changes",input:"bool(torch.allclose(q_learning_update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, 0, 5.0, 1, 0.5, 0.9), torch.tensor([[4.8, 2.0], [3.0, 4.0]])))",expected:"True",hidden:!0},{id:"4",description:"Terminal state (no future)",input:"bool(torch.allclose(q_learning_update(torch.zeros(2, 2), 0, 0, 10.0, 1, 1.0, 0.0), torch.tensor([[10.0, 0.0], [0.0, 0.0]])))",expected:"True",hidden:!0},{id:"5",description:"Input Q-table is not modified",input:"(lambda: (Q := torch.zeros(2, 2), _ := q_learning_update(Q, 0, 1, 1.0, 1, 0.1, 0.9), Q.tolist())[-1])()",expected:"[[0.0, 0.0], [0.0, 0.0]]",hidden:!0}],hints:["Best next value: Q[next_state].max()","TD target: reward + gamma * max_next_Q; TD error: target - Q[state, action]","Update: Q[state, action] += alpha * td_error (on a clone)"]},"rl-sarsa-update":{note:"Like Q-learning, but bootstrap from Q[next_state, next_action]. Work on Q.clone() and return the updated tensor.",starterCode:`import torch

def sarsa_update(Q: torch.Tensor, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float) -> torch.Tensor:
    """
    Perform one SARSA update.

    Args:
        Q: Q-table tensor of shape (n_states, n_actions)
        state: Current state s
        action: Action taken a
        reward: Reward received r
        next_state: Next state s'
        next_action: Next action a' (will be taken)
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        Updated Q-table (new tensor; the input is left unchanged)
    """
    Q = Q.clone().float()
    # Your code here
    pass
`,solution:`import torch

def sarsa_update(Q: torch.Tensor, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float) -> torch.Tensor:
    Q = Q.clone().float()

    # SARSA bootstraps from the action actually taken next (on-policy)
    td_target = reward + gamma * Q[next_state, next_action]
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error
    return Q
`,testCases:[{id:"1",description:"Basic SARSA update",input:"round(sarsa_update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, 0, 5.0, 1, 0, 0.5, 0.9)[0, 0].item(), 4)",expected:"4.35",hidden:!1},{id:"2",description:"Different next action",input:"bool(torch.allclose(sarsa_update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, 0, 5.0, 1, 1, 0.5, 0.9), torch.tensor([[4.8, 2.0], [3.0, 4.0]])))",expected:"True",hidden:!1},{id:"3",description:"From zero",input:"bool(torch.allclose(sarsa_update(torch.zeros(2, 2), 0, 1, 1.0, 1, 0, 0.1, 0.9), torch.tensor([[0.0, 0.1], [0.0, 0.0]])))",expected:"True",hidden:!0},{id:"4",description:"Uses Q(s', a'), not the max",input:"round(sarsa_update(torch.tensor([[0.0, 0.0], [10.0, -10.0]]), 0, 1, 0.0, 1, 1, 1.0, 1.0)[0, 1].item(), 4)",expected:"-10.0",hidden:!0}],hints:["The key difference from Q-learning: use Q[next_state, next_action] not the max","TD target: reward + gamma * Q[next_state, next_action]","Update: Q[state, action] += alpha * (target - Q[state, action])"]},"rl-td0-prediction":{note:"Index the value tensor directly. Work on V.clone() and return the updated tensor.",starterCode:`import torch

def td0_update(V: torch.Tensor, state: int, reward: float,
               next_state: int, alpha: float, gamma: float,
               done: bool = False) -> torch.Tensor:
    """
    Perform one TD(0) value update.

    Args:
        V: Value function tensor V(s) for each state
        state: Current state s
        reward: Reward received r
        next_state: Next state s'
        alpha: Learning rate
        gamma: Discount factor
        done: Whether next_state is terminal

    Returns:
        Updated value function (new tensor)
    """
    V = V.clone().float()
    # Your code here
    pass
`,solution:`import torch

def td0_update(V: torch.Tensor, state: int, reward: float,
               next_state: int, alpha: float, gamma: float,
               done: bool = False) -> torch.Tensor:
    V = V.clone().float()

    # Terminal next state has value 0
    next_value = 0.0 if done else V[next_state]
    td_error = reward + gamma * next_value - V[state]
    V[state] += alpha * td_error
    return V
`,testCases:[{id:"1",description:"Non-terminal transition",input:"bool(torch.allclose(td0_update(torch.zeros(3), 0, 1.0, 1, 0.1, 0.9, False), torch.tensor([0.1, 0.0, 0.0])))",expected:"True",hidden:!1},{id:"2",description:"Terminal state",input:"td0_update(torch.zeros(3), 0, 10.0, 1, 0.5, 0.9, True).tolist()",expected:"[5.0, 0.0, 0.0]",hidden:!1},{id:"3",description:"With existing values",input:"round(td0_update(torch.tensor([5.0, 10.0, 15.0]), 1, 2.0, 2, 0.1, 0.9, False)[1].item(), 4)",expected:"10.55",hidden:!0},{id:"4",description:"done=True ignores V(s')",input:"round(td0_update(torch.tensor([0.0, 100.0]), 0, 1.0, 1, 1.0, 0.9, True)[0].item(), 4)",expected:"1.0",hidden:!0}],hints:["If done=True, the next state value is 0 (terminal)","TD error: reward + gamma * V[next_state] - V[state]","Update: V[state] += alpha * td_error (on a clone)"]},"rl-value-iteration":{note:"Compute each action value with tensor ops and take the max with torch.stack(...).max(). Return the new value tensor.",starterCode:`import torch

def value_iteration_step(V: torch.Tensor, transitions: dict,
                         gamma: float) -> torch.Tensor:
    """
    Perform one step of value iteration.

    Args:
        V: Current value function tensor V(s) for each state
        transitions: Dict where transitions[s][a] = [(prob, reward, next_state), ...]
        gamma: Discount factor

    Returns:
        Updated value function (tensor)
    """
    V = V.float()
    n_states = V.shape[0]
    V_new = torch.zeros(n_states)

    # Your code here
    pass
`,solution:`import torch

def value_iteration_step(V: torch.Tensor, transitions: dict,
                         gamma: float) -> torch.Tensor:
    V = V.float()
    n_states = V.shape[0]
    V_new = torch.zeros(n_states)

    for s in range(n_states):
        if s not in transitions:
            continue

        q_values = []
        for outcomes in transitions[s].values():
            probs = torch.tensor([float(p) for p, _, _ in outcomes])
            rewards = torch.tensor([float(r) for _, r, _ in outcomes])
            next_states = torch.tensor([s2 for _, _, s2 in outcomes])
            q_values.append((probs * (rewards + gamma * V[next_states])).sum())

        # Bellman optimality: V(s) = max_a Q(s, a)
        if q_values:
            V_new[s] = torch.stack(q_values).max()

    return V_new
`,testCases:[{id:"1",description:"Simple 2-state MDP",input:"value_iteration_step(torch.tensor([0.0, 0.0]), {0: {0: [(1.0, 1.0, 0)], 1: [(1.0, 5.0, 1)]}, 1: {0: [(1.0, 2.0, 0)], 1: [(1.0, 0.0, 1)]}}, 0.9).tolist()",expected:"[5.0, 2.0]",hidden:!1},{id:"2",description:"With existing values",input:"bool(torch.allclose(value_iteration_step(torch.tensor([5.0, 2.0]), {0: {0: [(1.0, 1.0, 0)], 1: [(1.0, 5.0, 1)]}, 1: {0: [(1.0, 2.0, 0)], 1: [(1.0, 0.0, 1)]}}, 0.9), torch.tensor([6.8, 6.5])))",expected:"True",hidden:!1},{id:"3",description:"Stochastic transitions",input:"bool(torch.allclose(value_iteration_step(torch.tensor([1.0, 2.0, 0.0]), {0: {0: [(0.5, 1.0, 1), (0.5, 0.0, 2)], 1: [(1.0, 0.5, 0)]}, 1: {0: [(0.2, 3.0, 0), (0.8, 0.0, 1)]}, 2: {0: [(1.0, 0.0, 2)]}}, 0.5), torch.tensor([1.0, 1.5, 0.0])))",expected:"True",hidden:!0}],hints:["For each state, compute Q(s, a) = sum over outcomes of prob * (reward + gamma * V[next])","Take the max over actions: torch.stack(q_values).max()","Read from the old V, write into V_new"]},"rl-policy-gradient":{note:"Let autograd compute the gradient: build the REINFORCE objective sum_t G_t * log π(a_t|s_t) with F.log_softmax and call .backward() on a copy of theta that requires grad. (Deriving one_hot(a) - π by hand also works.)",starterCode:`import torch
import torch.nn.functional as F

def reinforce_gradient(theta: torch.Tensor, episode: list,
                       gamma: float) -> torch.Tensor:
    """
    Compute REINFORCE policy gradient for a tabular softmax policy.

    Args:
        theta: Policy parameters (n_states, n_actions)
        episode: List of (state, action, reward) tuples
        gamma: Discount factor

    Returns:
        Gradient of sum_t G_t * log π(a_t|s_t) w.r.t. theta, same shape as theta
    """
    theta = theta.detach().clone().float().requires_grad_(True)

    # Your code here
    # 1. Compute returns G_t for each timestep
    # 2. Objective = sum_t G_t * log softmax(theta[s_t])[a_t]
    # 3. objective.backward() and return theta.grad
    pass
`,solution:`import torch
import torch.nn.functional as F

def reinforce_gradient(theta: torch.Tensor, episode: list,
                       gamma: float) -> torch.Tensor:
    theta = theta.detach().clone().float().requires_grad_(True)

    T = len(episode)
    if T == 0:
        return torch.zeros_like(theta)

    states = torch.tensor([s for s, _, _ in episode])
    actions = torch.tensor([a for _, a, _ in episode])

    # Returns G_t, computed backwards
    returns = torch.zeros(T)
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = episode[t][2] + gamma * G
        returns[t] = G

    # log π(a_t | s_t) for the actions actually taken
    log_probs = F.log_softmax(theta[states], dim=-1)
    chosen = log_probs[torch.arange(T), actions]

    # ∇ sum_t G_t log π(a_t|s_t) = sum_t G_t (one_hot(a_t) - π(.|s_t))
    objective = (returns * chosen).sum()
    objective.backward()
    return theta.grad
`,testCases:[{id:"1",description:"Simple episode",input:"bool(torch.allclose(reinforce_gradient(torch.zeros(2, 2), [(0, 1, 1.0), (1, 0, 1.0)], 0.9), torch.tensor([[-0.95, 0.95], [0.5, -0.5]]), atol=0.01))",expected:"True",hidden:!1},{id:"2",description:"Single step episode",input:"bool(torch.allclose(reinforce_gradient(torch.zeros(2, 2), [(0, 0, 5.0)], 0.9), torch.tensor([[2.5, -2.5], [0.0, 0.0]]), atol=0.01))",expected:"True",hidden:!1},{id:"3",description:"Non-uniform policy: G * (one_hot(a) - softmax(theta[s]))",input:"bool(torch.allclose(reinforce_gradient(torch.tensor([[0.5, -0.5], [1.0, 0.0]]), [(0, 0, 2.0)], 0.9), torch.tensor([[0.5379, -0.5379], [0.0, 0.0]]), atol=1e-3))",expected:"True",hidden:!0},{id:"4",description:"Repeated state accumulates and the input is not modified",input:`(lambda: (
            theta := torch.tensor([[0.0, 1.0, -1.0]]),
            g := reinforce_gradient(theta, [(0, 2, 1.0), (0, 1, 0.0), (0, 0, 2.0)], 0.5),
            p := torch.softmax(theta[0], dim=0),
            ref := 1.5 * (torch.tensor([0.0, 0.0, 1.0]) - p) + 1.0 * (torch.tensor([0.0, 1.0, 0.0]) - p) + 2.0 * (torch.tensor([1.0, 0.0, 0.0]) - p),
            bool(torch.allclose(g[0], ref, atol=1e-4) and theta.grad is None and theta.tolist() == [[0.0, 1.0, -1.0]])
        )[-1])()`,expected:"True",hidden:!0}],hints:["Compute returns backwards: G_t = r_t + gamma * G_{t+1}","F.log_softmax(theta[states], dim=-1)[torch.arange(T), actions] gives log π(a_t|s_t)","objective = (returns * log_probs).sum(); objective.backward(); return theta.grad","By hand instead: theta.grad[s] += G_t * (one_hot(a) - softmax(theta[s]))"]},"rl-advantage-estimation":{note:"A backward loop over timesteps with tensor indexing. Return the advantages as a tensor.",starterCode:`import torch

def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                gamma: float, lam: float) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: Rewards r_0, r_1, ..., r_{T-1} (length T)
        values: Values V(s_0), V(s_1), ..., V(s_T) (length T+1)
                Last value is V(s_T) or 0 if terminal
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        Advantages Â_0, Â_1, ..., Â_{T-1} (tensor of length T)
    """
    rewards = rewards.float()
    values = values.float()
    T = rewards.shape[0]
    advantages = torch.zeros(T)

    # Your code here
    pass
`,solution:`import torch

def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                gamma: float, lam: float) -> torch.Tensor:
    rewards = rewards.float()
    values = values.float()
    T = rewards.shape[0]
    advantages = torch.zeros(T)

    gae = 0.0
    for t in range(T - 1, -1, -1):
        # TD error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        # Â_t = δ_t + γλ Â_{t+1}
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    return advantages
`,testCases:[{id:"1",description:"Zero values baseline",input:"bool(torch.allclose(compute_gae(torch.tensor([1.0, 1.0, 1.0]), torch.zeros(4), 0.99, 0.95), torch.tensor([2.8250, 1.9405, 1.0]), atol=1e-3))",expected:"True",hidden:!1},{id:"2",description:"With value estimates",input:"bool(torch.allclose(compute_gae(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0, 0.0]), 0.9, 0.9), torch.tensor([3.987, 2.7, 0.0]), atol=1e-3))",expected:"True",hidden:!1},{id:"3",description:"Single step",input:"bool(torch.allclose(compute_gae(torch.tensor([5.0]), torch.tensor([1.0, 0.0]), 0.99, 0.95), torch.tensor([4.0]), atol=1e-4))",expected:"True",hidden:!0},{id:"4",description:"lambda = 1: advantages + values = discounted returns",input:`(lambda: (
            r := torch.tensor([1.0, -1.0, 2.0, 0.5]), v := torch.tensor([0.3, -0.2, 0.7, 1.1, 0.0]),
            ret := torch.tensor([1.0 - 0.9 + 0.81 * 2.0 + 0.729 * 0.5, -1.0 + 0.9 * 2.0 + 0.81 * 0.5, 2.0 + 0.9 * 0.5, 0.5]),
            bool(torch.allclose(compute_gae(r, v, 0.9, 1.0) + v[:-1], ret, atol=1e-4))
        )[-1])()`,expected:"True",hidden:!0}],hints:["TD errors: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)","Work backwards from the last timestep: Â_t = δ_t + γλ*Â_{t+1}","Store each Â_t with advantages[t] = gae"]},"rl-ppo-clip":{note:"Use torch.clamp and torch.minimum. Return the objective as a Python float.",starterCode:`import torch

def ppo_clip_objective(old_probs: torch.Tensor, new_probs: torch.Tensor,
                       advantages: torch.Tensor, epsilon: float) -> float:
    """
    Compute PPO clipped surrogate objective.

    Args:
        old_probs: π_old(a|s) probabilities for taken actions
        new_probs: π_new(a|s) probabilities for taken actions
        advantages: Advantage estimates A_t
        epsilon: Clipping parameter (e.g., 0.2)

    Returns:
        PPO clipped objective (Python float rounded to 2 decimals, should be maximized)
    """
    old_probs = old_probs.double()
    new_probs = new_probs.double()
    advantages = advantages.double()

    # Your code here
    pass
`,solution:`import torch

def ppo_clip_objective(old_probs: torch.Tensor, new_probs: torch.Tensor,
                       advantages: torch.Tensor, epsilon: float) -> float:
    old_probs = old_probs.double()
    new_probs = new_probs.double()
    advantages = advantages.double()

    ratios = new_probs / (old_probs + 1e-8)
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)

    # Pessimistic (minimum) of the unclipped and clipped surrogates
    objective = torch.minimum(ratios * advantages, clipped_ratios * advantages).mean()
    return round(objective.item(), 2)
`,testCases:[{id:"1",description:"No clipping needed",input:"ppo_clip_objective(torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5]), torch.tensor([1.0, -1.0]), 0.2)",expected:"0.0",hidden:!1},{id:"2",description:"Positive advantage, ratio > 1",input:"ppo_clip_objective(torch.tensor([0.5]), torch.tensor([0.7]), torch.tensor([1.0]), 0.2)",expected:"1.2",hidden:!1},{id:"3",description:"Negative advantage, ratio > 1",input:"ppo_clip_objective(torch.tensor([0.5]), torch.tensor([0.7]), torch.tensor([-1.0]), 0.2)",expected:"-1.4",hidden:!1},{id:"4",description:"Clipping in action",input:"ppo_clip_objective(torch.tensor([0.3]), torch.tensor([0.6]), torch.tensor([2.0]), 0.2)",expected:"2.4",hidden:!0},{id:"5",description:"Negative advantage, ratio < 1 is clipped at 1 - epsilon",input:"ppo_clip_objective(torch.tensor([0.5, 0.4]), torch.tensor([0.2, 0.4]), torch.tensor([-2.0, 1.0]), 0.2)",expected:"-0.3",hidden:!0}],hints:["ratios = new_probs / old_probs","clipped = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)","objective = torch.minimum(ratios * A, clipped * A).mean()"]},"rl-n-step-return":{note:"Inputs are tensors; return a Python float.",starterCode:`import torch

def n_step_return(rewards: torch.Tensor, values: torch.Tensor,
                  t: int, n: int, gamma: float) -> float:
    """
    Compute n-step return from timestep t.

    Args:
        rewards: All rewards [r_0, r_1, ..., r_{T-1}]
        values: All values [V(s_0), ..., V(s_T)]
        t: Starting timestep
        n: Number of steps
        gamma: Discount factor

    Returns:
        N-step return G_t^(n) (Python float rounded to 2 decimals)
    """
    rewards = rewards.double()
    values = values.double()
    T = rewards.shape[0]

    # Your code here
    pass
`,solution:`import torch

def n_step_return(rewards: torch.Tensor, values: torch.Tensor,
                  t: int, n: int, gamma: float) -> float:
    rewards = rewards.double()
    values = values.double()
    T = rewards.shape[0]

    # Discounted rewards r_t ... r_{t+n-1} (truncated at the end of the episode)
    end = min(t + n, T)
    discounts = gamma ** torch.arange(end - t, dtype=torch.float64)
    G = (discounts * rewards[t:end]).sum().item()

    # Bootstrap from V(s_{t+n}) if it exists
    if t + n < values.shape[0]:
        G += (gamma ** n) * values[t + n].item()

    return round(G, 2)
`,testCases:[{id:"1",description:"2-step return",input:"n_step_return(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0]), 0, 2, 0.9)",expected:"2.8",hidden:!1},{id:"2",description:"3-step return with bootstrap",input:"n_step_return(torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0]), 0, 3, 0.9)",expected:"6.36",hidden:!1},{id:"3",description:"1-step (TD) return",input:"n_step_return(torch.tensor([5.0, 2.0, 3.0]), torch.tensor([1.0, 10.0, 5.0, 0.0]), 0, 1, 0.9)",expected:"14.0",hidden:!0},{id:"4",description:"Starting mid-episode, n past the end",input:"n_step_return(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.0, 0.0, 0.0, 0.0]), 1, 5, 0.5)",expected:"3.5",hidden:!0}],hints:["Sum gamma**k * rewards[t + k] for k < n while t + k < T","Bootstrap with gamma**n * values[t + n] if t + n < len(values)","Return round(G, 2) as a Python float"]}},Zx={"rl-reward-modeling":{note:"Use torch.sigmoid and torch.log. Don't use F.binary_cross_entropy_with_logits (it computes exactly this loss when the target is 1); a hidden test uses it as the reference.",starterCode:`import torch

def reward_model_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float:
    """
    Compute RLHF reward model loss (Bradley-Terry).

    Args:
        r_chosen: Reward scores for preferred/chosen responses
        r_rejected: Reward scores for rejected responses

    Returns:
        Average negative log-likelihood loss (Python float rounded to 3 decimals)
    """
    r_chosen = r_chosen.double()
    r_rejected = r_rejected.double()

    # Your code here
    # Loss = -E[log σ(r_chosen - r_rejected)]
    pass
`,solution:`import torch

def reward_model_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float:
    r_chosen = r_chosen.double()
    r_rejected = r_rejected.double()

    # P(chosen > rejected) = σ(r_chosen - r_rejected)
    logits = r_chosen - r_rejected
    loss = -torch.log(torch.sigmoid(logits)).mean()
    return round(loss.item(), 3)
`,testCases:[{id:"1",description:"Clear preference",input:"reward_model_loss(torch.tensor([2.0, 1.5]), torch.tensor([1.0, 0.5]))",expected:"0.313",hidden:!1},{id:"2",description:"Equal scores",input:"reward_model_loss(torch.tensor([0.0]), torch.tensor([0.0]))",expected:"0.693",hidden:!1},{id:"3",description:"Large margin",input:"reward_model_loss(torch.tensor([5.0, 4.0, 3.0]), torch.tensor([0.0, 0.0, 0.0]))",expected:"0.024",hidden:!0},{id:"4",description:"Matches BCE-with-logits against target 1",input:`(lambda: (
            c := torch.randn(8), r := torch.randn(8),
            bool(abs(reward_model_loss(c, r) - torch.nn.functional.binary_cross_entropy_with_logits(c - r, torch.ones(8)).item()) < 1e-3)
        )[-1])()`,expected:"True",hidden:!0}],hints:["Compute the difference: r_chosen - r_rejected","torch.sigmoid gives the preference probability","Loss: -torch.log(probs).mean()"]},"rl-dpo-loss":{note:"Pure tensor arithmetic with torch.sigmoid and torch.log. Return a Python float.",starterCode:`import torch

def dpo_loss(logp_chosen: torch.Tensor, logp_rejected: torch.Tensor,
             ref_logp_chosen: torch.Tensor, ref_logp_rejected: torch.Tensor,
             beta: float) -> float:
    """
    Compute Direct Preference Optimization (DPO) loss.

    Args:
        logp_chosen: Log prob of chosen response under policy π_θ
        logp_rejected: Log prob of rejected response under policy π_θ
        ref_logp_chosen: Log prob of chosen under reference π_ref
        ref_logp_rejected: Log prob of rejected under reference π_ref
        beta: Temperature parameter

    Returns:
        DPO loss (Python float rounded to 3 decimals)
    """
    logp_chosen = logp_chosen.double()
    logp_rejected = logp_rejected.double()
    ref_logp_chosen = ref_logp_chosen.double()
    ref_logp_rejected = ref_logp_rejected.double()

    # Your code here
    pass
`,solution:`import torch

def dpo_loss(logp_chosen: torch.Tensor, logp_rejected: torch.Tensor,
             ref_logp_chosen: torch.Tensor, ref_logp_rejected: torch.Tensor,
             beta: float) -> float:
    logp_chosen = logp_chosen.double()
    logp_rejected = logp_rejected.double()
    ref_logp_chosen = ref_logp_chosen.double()
    ref_logp_rejected = ref_logp_rejected.double()

    # Log-ratios of policy vs reference
    chosen_logratios = logp_chosen - ref_logp_chosen
    rejected_logratios = logp_rejected - ref_logp_rejected

    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -torch.log(torch.sigmoid(logits)).mean()
    return round(loss.item(), 3)
`,testCases:[{id:"1",description:"Policy prefers chosen",input:"dpo_loss(torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-1.5]), torch.tensor([-1.5]), 0.1)",expected:"0.644",hidden:!1},{id:"2",description:"Equal preferences",input:"dpo_loss(torch.tensor([-1.0, -1.0]), torch.tensor([-1.0, -1.0]), torch.tensor([-1.0, -1.0]), torch.tensor([-1.0, -1.0]), 0.5)",expected:"0.693",hidden:!1},{id:"3",description:"Strong preference signal",input:"dpo_loss(torch.tensor([0.0]), torch.tensor([-5.0]), torch.tensor([-2.0]), torch.tensor([-3.0]), 0.1)",expected:"0.513",hidden:!0},{id:"4",description:"Policy prefers the rejected response: loss above log 2",input:"dpo_loss(torch.tensor([-3.0, -2.0]), torch.tensor([-1.0, -1.0]), torch.tensor([-2.0, -2.0]), torch.tensor([-2.0, -2.0]), 0.5)",expected:"1.144",hidden:!0}],hints:["Δ_w = logp_chosen - ref_logp_chosen, Δ_l = logp_rejected - ref_logp_rejected","logits = beta * (Δ_w - Δ_l)","Loss: -torch.log(torch.sigmoid(logits)).mean()"]},"rl-kl-penalty":{note:"One line of tensor arithmetic. Return a Python float.",starterCode:`import torch

def kl_penalty(logp_policy: torch.Tensor, logp_reference: torch.Tensor,
               beta: float) -> float:
    """
    Compute KL divergence penalty for RLHF.

    Args:
        logp_policy: Log probs from current policy π_θ
        logp_reference: Log probs from reference policy π_ref
        beta: KL penalty coefficient

    Returns:
        KL penalty = β * mean(KL divergence) (Python float rounded to 2 decimals)
    """
    logp_policy = logp_policy.double()
    logp_reference = logp_reference.double()

    # Your code here
    pass
`,solution:`import torch

def kl_penalty(logp_policy: torch.Tensor, logp_reference: torch.Tensor,
               beta: float) -> float:
    logp_policy = logp_policy.double()
    logp_reference = logp_reference.double()

    # Per-token KL estimate: log π_θ - log π_ref
    mean_kl = (logp_policy - logp_reference).mean()
    return round((beta * mean_kl).item(), 2)
`,testCases:[{id:"1",description:"Positive divergence",input:"kl_penalty(torch.tensor([-1.0, -2.0]), torch.tensor([-1.5, -2.5]), 0.1)",expected:"0.05",hidden:!1},{id:"2",description:"No divergence",input:"kl_penalty(torch.tensor([-1.0, -1.0]), torch.tensor([-1.0, -1.0]), 0.1)",expected:"0.0",hidden:!1},{id:"3",description:"Higher beta",input:"kl_penalty(torch.tensor([-0.5, -1.0, -1.5]), torch.tensor([-1.0, -1.5, -2.0]), 0.5)",expected:"0.25",hidden:!0},{id:"4",description:"Mean over tokens of mixed-sign differences",input:"kl_penalty(torch.tensor([-1.0, -3.0, -0.5, -2.0]), torch.tensor([-2.0, -2.0, -1.5, -2.6]), 1.0)",expected:"0.4",hidden:!0}],hints:["Per-token KL: logp_policy - logp_reference","Take the mean over tokens and multiply by beta","Return round(penalty.item(), 2)"]},"rl-ppo-llm-objective":{note:"Use torch.exp, torch.clamp and torch.minimum. Return a dict of Python floats.",starterCode:`import torch

def ppo_rlhf_objective(old_logp: torch.Tensor, new_logp: torch.Tensor,
                       ref_logp: torch.Tensor, rewards: torch.Tensor,
                       epsilon: float, beta: float) -> dict:
    """
    Compute PPO-RLHF objective for LLM fine-tuning.

    Args:
        old_logp: Log probs from old policy (before update)
        new_logp: Log probs from current policy
        ref_logp: Log probs from reference (pretrained) model
        rewards: Rewards from reward model
        epsilon: PPO clipping parameter
        beta: KL penalty coefficient

    Returns:
        Dictionary with 'objective', 'policy_loss', 'kl_penalty' (Python floats rounded to 2 decimals)
    """
    old_logp = old_logp.double()
    new_logp = new_logp.double()
    ref_logp = ref_logp.double()
    rewards = rewards.double()

    # Your code here
    pass
`,solution:`import torch

def ppo_rlhf_objective(old_logp: torch.Tensor, new_logp: torch.Tensor,
                       ref_logp: torch.Tensor, rewards: torch.Tensor,
                       epsilon: float, beta: float) -> dict:
    old_logp = old_logp.double()
    new_logp = new_logp.double()
    ref_logp = ref_logp.double()
    rewards = rewards.double()

    # Probability ratio π_θ / π_old from log probs
    ratio = torch.exp(new_logp - old_logp)

    # KL from the reference model, and the penalty
    kl = new_logp - ref_logp
    kl_penalty = beta * kl.mean()

    # RLHF advantage: reward minus per-token KL penalty
    advantage = rewards - beta * kl

    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()

    objective = policy_loss

    return {
        'objective': round(objective.item(), 2),
        'policy_loss': round(policy_loss.item(), 2),
        'kl_penalty': round(kl_penalty.item(), 2),
    }
`,testCases:[{id:"1",description:"Positive reward, small update",input:"(lambda r: bool(abs(r['objective'] - 1.18) < 0.1 and abs(r['kl_penalty'] - 0.02) < 0.01))(ppo_rlhf_objective(torch.tensor([-2.0]), torch.tensor([-1.8]), torch.tensor([-2.0]), torch.tensor([1.0]), 0.2, 0.1))",expected:"True",hidden:!1},{id:"2",description:"No policy change",input:"(lambda r: bool(abs(r['objective'] - 1.0) < 0.01 and abs(r['kl_penalty']) < 0.01))(ppo_rlhf_objective(torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([1.0]), 0.2, 0.1))",expected:"True",hidden:!1},{id:"3",description:"Clipped ratio and KL penalty values",input:"(lambda r: (r['policy_loss'], r['kl_penalty']))(ppo_rlhf_objective(torch.tensor([-2.0, -1.0]), torch.tensor([-1.0, -1.0]), torch.tensor([-1.5, -2.0]), torch.tensor([2.0, 1.0]), 0.2, 0.2))",expected:"(1.54, 0.15)",hidden:!0}],hints:["ratio = torch.exp(new_logp - old_logp)","kl = new_logp - ref_logp; kl_penalty = beta * kl.mean()","advantage = rewards - beta * kl","policy_loss = torch.minimum(ratio * A, torch.clamp(ratio, 1 - eps, 1 + eps) * A).mean()"]}},Jx={"e2e-mlp":{note:"MLP is an nn.Module with nn.Parameter weights. Write backward() by hand (no loss.backward() or F.cross_entropy inside the class); a hidden test checks your gradients against autograd. compute_loss returns a scalar tensor.",starterCode:`import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize 2-layer MLP with random weights."""
        super().__init__()
        torch.manual_seed(42)
        # He initialization
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * math.sqrt(2.0 / input_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim) * math.sqrt(2.0 / hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(output_dim))

        # Cache for backward pass
        self.cache = {}

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Input (batch, input_dim)

        Returns:
            logits: Output (batch, output_dim)
        """
        # Your code here
        pass

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss (and cache the softmax probabilities).

        Args:
            logits: Model output (batch, output_dim)
            y: True labels (batch,) as int64

        Returns:
            loss: Scalar tensor (call .item() for a float)
        """
        # Your code here
        pass

    def backward(self, y: torch.Tensor) -> dict:
        """
        Backward pass, derived by hand from the cached forward values.

        Args:
            y: True labels (batch,) as int64

        Returns:
            gradients: Dictionary with dW1, db1, dW2, db2 (mean loss over the batch)
        """
        # Your code here
        pass

    def train_step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 0.01) -> float:
        """
        Single training step: forward, loss, backward, update.

        Returns:
            loss: Loss value before update (Python float)
        """
        # Your code here
        pass


def train_mlp(X_train, y_train, epochs=100, lr=0.01):
    """Train MLP (hidden_dim=64) and return (model, loss history)."""
    # Your code here
    pass
`,solution:`import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        torch.manual_seed(42)
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * math.sqrt(2.0 / input_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim) * math.sqrt(2.0 / hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(output_dim))
        self.cache = {}

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z1 = X @ self.W1 + self.b1
        h = torch.relu(z1)
        logits = h @ self.W2 + self.b2
        self.cache = {'X': X, 'z1': z1, 'h': h, 'logits': logits}
        return logits

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Numerically stable log-softmax
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        self.cache['probs'] = log_probs.exp()
        return -log_probs[torch.arange(logits.shape[0]), y].mean()

    def backward(self, y: torch.Tensor) -> dict:
        with torch.no_grad():
            X = self.cache['X'].detach()
            z1 = self.cache['z1'].detach()
            h = self.cache['h'].detach()
            probs = self.cache['probs'].detach()
            batch_size = X.shape[0]

            y_onehot = F.one_hot(y, probs.shape[1]).float()

            # Output layer (softmax + cross-entropy, mean over the batch)
            d_logits = (probs - y_onehot) / batch_size
            dW2 = h.T @ d_logits
            db2 = d_logits.sum(dim=0)

            # Hidden layer
            d_h = d_logits @ self.W2.T
            d_relu = d_h * (z1 > 0).float()
            dW1 = X.T @ d_relu
            db1 = d_relu.sum(dim=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def train_step(self, X: torch.Tensor, y: torch.Tensor, lr: float = 0.01) -> float:
        logits = self.forward(X)
        loss = self.compute_loss(logits, y)
        grads = self.backward(y)

        # Plain SGD update on the parameters (no autograd tracking)
        with torch.no_grad():
            self.W1 -= lr * grads['dW1']
            self.b1 -= lr * grads['db1']
            self.W2 -= lr * grads['dW2']
            self.b2 -= lr * grads['db2']

        return loss.item()


def train_mlp(X_train, y_train, epochs=100, lr=0.01):
    input_dim = X_train.shape[1]
    output_dim = len(torch.unique(y_train))
    model = MLP(input_dim, 64, output_dim)

    losses = []
    for _ in range(epochs):
        losses.append(model.train_step(X_train, y_train, lr))
    return model, losses
`,testCases:[{id:"1",description:"Forward pass shape",input:"MLP(784, 128, 10).forward(torch.randn(4, 784)).shape",expected:"(4, 10)",hidden:!1},{id:"2",description:"Loss decreases",input:"(lambda: (m := MLP(10, 8, 3), X := torch.randn(16, 10), y := torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]), losses := [m.train_step(X, y) for _ in range(50)], bool(losses[-1] < losses[0]))[-1])()",expected:"True",hidden:!1},{id:"3",description:"Loss matches the cross-entropy of the logits",input:"(lambda: (m := MLP(10, 8, 3), X := torch.randn(4, 10), y := torch.tensor([0, 1, 2, 0]), logits := m.forward(X), bool(abs(float(m.compute_loss(logits, y)) - torch.nn.functional.cross_entropy(logits, y).item()) < 1e-5))[-1])()",expected:"True",hidden:!1},{id:"4",description:"Gradients computed for every parameter",input:"(lambda: (m := MLP(10, 8, 3), X := torch.randn(4, 10), y := torch.tensor([0, 1, 2, 0]), _ := m.compute_loss(m.forward(X), y), g := m.backward(y), sorted(g.keys()))[-1])()",expected:'["dW1", "dW2", "db1", "db2"]',hidden:!0},{id:"5",description:"Manual gradients match autograd",input:`(lambda: (
            m := MLP(10, 8, 3), X := torch.randn(6, 10), y := torch.tensor([0, 1, 2, 0, 2, 1]),
            loss := m.compute_loss(m.forward(X), y),
            g := m.backward(y),
            loss.backward(),
            bool(torch.allclose(g['dW1'], m.W1.grad, atol=1e-5) and torch.allclose(g['db1'], m.b1.grad, atol=1e-5)
                 and torch.allclose(g['dW2'], m.W2.grad, atol=1e-5) and torch.allclose(g['db2'], m.b2.grad, atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0},{id:"6",description:"train_step applies W -= lr * dW",input:`(lambda: (
            m := MLP(5, 4, 2), X := torch.randn(3, 5), y := torch.tensor([0, 1, 1]),
            W2_before := m.W2.detach().clone(),
            _ := m.compute_loss(m.forward(X), y), g := m.backward(y),
            _ := m.train_step(X, y, lr=0.5),
            bool(isinstance(m, torch.nn.Module) and torch.allclose(m.W2.detach(), W2_before - 0.5 * g['dW2'], atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0},{id:"7",description:"train_mlp returns the model and a decreasing loss history",input:"(lambda: (r := train_mlp(torch.randn(20, 5), torch.tensor([0, 1] * 10), epochs=20, lr=0.1), bool(len(r[1]) == 20 and r[1][-1] < r[1][0]))[-1])()",expected:"True",hidden:!0}],hints:["Cache X, z1 (pre-activation), h and logits in forward","Stable log-softmax: logits - torch.logsumexp(logits, dim=-1, keepdim=True)","d_logits = (probs - one_hot(y)) / batch_size; dW2 = h.T @ d_logits; d_relu = (d_logits @ W2.T) * (z1 > 0)","Update parameters inside torch.no_grad(): self.W1 -= lr * dW1"]},"e2e-transformer":{note:"TransformerEncoderBlock is an nn.Module with nn.Parameter weights. Implement attention and layer norm with tensor ops: don't use nn.MultiheadAttention, F.scaled_dot_product_attention, F.layer_norm/nn.LayerNorm or nn.TransformerEncoderLayer. Hidden tests compare each piece against those built-ins.",starterCode:`import math
import torch
import torch.nn as nn


def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization over the last dimension."""
    # Your code here
    pass


def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding (seq_len, d_model)."""
    # Your code here
    pass


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention over the last two dims.
    mask (optional) is additive (0 / -inf) and is added to the scores.

    Returns:
        (output, attention_weights)
    """
    # Your code here
    pass


def multi_head_attention(Q, K, V, num_heads):
    """Multi-head attention on (batch, seq_len, d_model) inputs, no projections."""
    # Your code here
    pass


def feed_forward(x, W1, b1, W2, b2):
    """Position-wise feed-forward network: Linear -> ReLU -> Linear"""
    # Your code here
    pass


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """
        Initialize transformer encoder block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
        """
        super().__init__()
        torch.manual_seed(42)
        self.d_model = d_model
        self.num_heads = num_heads

        # Initialize weights as nn.Parameter
        # Your code here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder block (post-norm).

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Your code here
        pass


def transformer_encoder(x, num_layers=2, d_model=64, num_heads=8, d_ff=256):
    """Add positional encoding, then stack multiple encoder blocks."""
    # Your code here
    pass
`,solution:`import math
import torch
import torch.nn as nn


def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)
    return PE


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attention = softmax(scores, dim=-1)
    return attention @ V, attention


def multi_head_attention(Q, K, V, num_heads):
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
    def split_heads(x):
        return x.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)

    out, _ = scaled_dot_product_attention(split_heads(Q), split_heads(K), split_heads(V))

    # Concatenate heads back to (batch, seq_len, d_model)
    return out.transpose(1, 2).reshape(batch, seq_len, d_model)


def feed_forward(x, W1, b1, W2, b2):
    return torch.relu(x @ W1 + b1) @ W2 + b2


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        torch.manual_seed(42)
        self.d_model = d_model
        self.num_heads = num_heads

        # Layer norm parameters
        self.ln1_gamma = nn.Parameter(torch.ones(d_model))
        self.ln1_beta = nn.Parameter(torch.zeros(d_model))
        self.ln2_gamma = nn.Parameter(torch.ones(d_model))
        self.ln2_beta = nn.Parameter(torch.zeros(d_model))

        # FFN parameters
        self.W1 = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.W2 = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention sublayer + residual + layer norm
        attn_output = multi_head_attention(x, x, x, self.num_heads)
        x = layer_norm(x + attn_output, self.ln1_gamma, self.ln1_beta)

        # Feed-forward sublayer + residual + layer norm
        ff_output = feed_forward(x, self.W1, self.b1, self.W2, self.b2)
        return layer_norm(x + ff_output, self.ln2_gamma, self.ln2_beta)


def transformer_encoder(x, num_layers=2, d_model=64, num_heads=8, d_ff=256):
    seq_len = x.shape[1]
    x = x + positional_encoding(seq_len, d_model)
    for _ in range(num_layers):
        x = TransformerEncoderBlock(d_model, num_heads, d_ff)(x)
    return x
`,testCases:[{id:"1",description:"Output shape preserved",input:"transformer_encoder(torch.randn(2, 10, 64), num_layers=2, d_model=64, num_heads=8, d_ff=256).shape",expected:"(2, 10, 64)",hidden:!1},{id:"2",description:"Attention weights sum to 1",input:"(lambda: (Q := torch.randn(2, 5, 64), K := torch.randn(2, 5, 64), V := torch.randn(2, 5, 64), out_attn := scaled_dot_product_attention(Q, K, V), bool(torch.allclose(out_attn[1].sum(dim=-1), torch.ones(2, 5))))[-1])()",expected:"True",hidden:!1},{id:"3",description:"Encoder output is layer-normalized (mean 0, variance 1 per position)",input:`(lambda: (
            out := transformer_encoder(torch.randn(2, 6, 16), num_layers=2, d_model=16, num_heads=4, d_ff=32),
            bool(torch.allclose(out.mean(dim=-1), torch.zeros(2, 6), atol=1e-4) and torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(2, 6), atol=1e-3))
        )[-1])()`,expected:"True",hidden:!1},{id:"4",description:"Attention (with an additive causal mask) matches F.scaled_dot_product_attention",input:`(lambda: (
            Q := torch.randn(2, 5, 8), K := torch.randn(2, 5, 8), V := torch.randn(2, 5, 8),
            mask := torch.triu(torch.full((5, 5), float('-inf')), diagonal=1),
            bool(torch.allclose(scaled_dot_product_attention(Q, K, V, mask)[0], torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask), atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0},{id:"5",description:"layer_norm matches F.layer_norm",input:`(lambda: (
            x := torch.randn(2, 3, 8) * 3, g := torch.randn(8), b := torch.randn(8),
            bool(torch.allclose(layer_norm(x, g, b), torch.nn.functional.layer_norm(x, (8,), g, b), atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0},{id:"6",description:"multi_head_attention matches per-head attention",input:`(lambda: (
            Q := torch.randn(2, 5, 16), K := torch.randn(2, 5, 16), V := torch.randn(2, 5, 16),
            heads := lambda t: t.reshape(2, 5, 4, 4).transpose(1, 2),
            ref := torch.nn.functional.scaled_dot_product_attention(heads(Q), heads(K), heads(V)).transpose(1, 2).reshape(2, 5, 16),
            bool(torch.allclose(multi_head_attention(Q, K, V, 4), ref, atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0},{id:"7",description:"Block is an nn.Module with registered parameters; feed_forward is Linear-ReLU-Linear",input:`(lambda: (
            blk := TransformerEncoderBlock(8, 2, 16),
            x := torch.tensor([[1.0, -2.0]]), W1 := torch.tensor([[1.0, 0.0], [0.0, 1.0]]), W2 := torch.tensor([[2.0], [3.0]]),
            bool(isinstance(blk, torch.nn.Module) and len(list(blk.parameters())) >= 4
                 and torch.allclose(feed_forward(x, W1, torch.zeros(2), W2, torch.tensor([0.5])), torch.tensor([[2.5]])))
        )[-1])()`,expected:"True",hidden:!0},{id:"8",description:"Positional encoding values",input:"bool(torch.allclose(positional_encoding(3, 4)[2], torch.tensor([0.909297, -0.416147, 0.019999, 0.9998]), atol=1e-5))",expected:"True",hidden:!0}],hints:["Attention: softmax(Q @ K.transpose(-2, -1) / sqrt(d_k) + mask) @ V","Heads: reshape to (batch, seq, heads, d_k) and transpose(1, 2); undo it after attention","Layer norm: (x - mean) / sqrt(var(unbiased=False) + eps) * gamma + beta over dim=-1","Register weights as nn.Parameter in __init__ (after super().__init__())","Post-norm block: x = LN(x + MHA(x)); x = LN(x + FFN(x))"]},"e2e-vae":{note:"VAE is an nn.Module built from nn.Linear layers. Write the reparameterization trick and the KL term yourself (no torch.distributions). compute_loss returns tensors so you can call total_loss.backward().",starterCode:`import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Initialize VAE.

        Args:
            input_dim: Input dimension (e.g., 784 for MNIST)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
        """
        super().__init__()
        torch.manual_seed(42)

        # Encoder
        self.enc = nn.Linear(input_dim, hidden_dim)

        # Latent space (mean and log variance)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)

        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encode input to latent distribution parameters.

        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        # Your code here
        pass

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        # Your code here
        pass

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction (sigmoid output in [0, 1]).
        """
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass.

        Returns:
            Dictionary with 'mu', 'log_var', 'z', 'x_recon'
        """
        # Your code here
        pass

    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                     mu: torch.Tensor, log_var: torch.Tensor) -> dict:
        """
        Compute VAE loss (ELBO): MSE reconstruction + KL (both averaged over all elements).

        Returns:
            Dictionary with 'total_loss', 'recon_loss', 'kl_loss' (scalar tensors)
        """
        # Your code here
        pass

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples by decoding random latent vectors.
        """
        # Your code here
        pass
`,solution:`import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        torch.manual_seed(42)
        self.enc = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple:
        h = torch.relu(self.enc(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.dec1(z))
        return torch.sigmoid(self.dec2(h))

    def forward(self, x: torch.Tensor) -> dict:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return {'mu': mu, 'log_var': log_var, 'z': z, 'x_recon': x_recon}

    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                     mu: torch.Tensor, log_var: torch.Tensor) -> dict:
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + kl_loss
        return {'total_loss': total_loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}

    def sample(self, num_samples: int) -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            return self.decode(z)
`,testCases:[{id:"1",description:"Reconstruction shape",input:'VAE(784, 256, 20).forward(torch.randn(4, 784))["x_recon"].shape',expected:"(4, 784)",hidden:!1},{id:"2",description:"KL divergence non-negative",input:'(lambda: (vae := VAE(784, 256, 20), x := torch.randn(4, 784), out := vae.forward(x), loss := vae.compute_loss(x, out["x_recon"], out["mu"], out["log_var"]), bool(loss["kl_loss"] >= 0))[-1])()',expected:"True",hidden:!1},{id:"3",description:"Loss values for fixed inputs",input:`(lambda: (
            loss := VAE(4, 8, 2).compute_loss(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.5, 0.5]]), torch.ones(1, 2), torch.zeros(1, 2)),
            (round(float(loss["recon_loss"]), 4), round(float(loss["kl_loss"]), 4), round(float(loss["total_loss"]), 4))
        )[-1])()`,expected:"(0.25, 0.5, 0.75)",hidden:!1},{id:"4",description:"Forward uses the reparameterization trick: z = mu + exp(0.5 * log_var) * eps",input:`(lambda: (
            vae := VAE(6, 8, 3),
            out := vae.forward(torch.randn(5, 6)),
            eps := (out["z"] - out["mu"]) / torch.exp(0.5 * out["log_var"]),
            bool(out["z"].shape == (5, 3) and abs(eps.mean().item()) < 1.5 and not torch.allclose(out["z"], out["mu"]))
        )[-1])()`,expected:"True",hidden:!0},{id:"5",description:"Gradients reach every parameter through the sample",input:`(lambda: (
            vae := VAE(10, 8, 2), x := torch.rand(4, 10),
            out := vae(x),
            loss := vae.compute_loss(x, out["x_recon"], out["mu"], out["log_var"]),
            loss["total_loss"].backward(),
            bool(isinstance(vae, torch.nn.Module) and all(p.grad is not None for p in vae.parameters()))
        )[-1])()`,expected:"True",hidden:!0},{id:"6",description:"Samples have the right shape and lie in [0, 1]",input:"(lambda: (s := VAE(12, 8, 3).sample(5), bool(s.shape == (5, 12) and s.min().item() >= 0 and s.max().item() <= 1))[-1])()",expected:"True",hidden:!0}],hints:["Encoder: h = relu(enc(x)); mu = fc_mu(h); log_var = fc_logvar(h)","Reparameterize: std = torch.exp(0.5 * log_var); z = mu + std * torch.randn_like(std)","Decoder: sigmoid(dec2(relu(dec1(z))))","KL = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp()); return tensors so total_loss.backward() works"]},"e2e-vqvae":{note:"Both classes are nn.Modules; the codebook is an nn.Parameter. Use .detach() for stop-gradients and the straight-through estimator z_e + (z_q - z_e).detach() in forward. Don't use torch.cdist.",starterCode:`import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Vector Quantization layer.

        Args:
            num_embeddings: Size of codebook (K)
            embedding_dim: Dimension of each embedding
        """
        super().__init__()
        torch.manual_seed(42)
        # Codebook with small random values, learnable
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.1)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def quantize(self, z_e: torch.Tensor) -> tuple:
        """
        Quantize encoder output to nearest codebook entries.

        Args:
            z_e: Encoder output (batch, height, width, embedding_dim)

        Returns:
            z_q: Quantized vectors (same shape as z_e), rows of the codebook
            indices: Codebook indices (batch, height, width), int64
            distances: Squared distance to the nearest embedding (batch, height, width)
        """
        # Your code here
        pass

    def compute_loss(self, z_e: torch.Tensor, z_q: torch.Tensor, beta: float = 0.25) -> dict:
        """
        Compute VQ losses.

        Returns:
            Dictionary with 'codebook_loss' (||sg[z_e] - z_q||^2) and
            'commitment_loss' (beta * ||z_e - sg[z_q]||^2), scalar tensors
        """
        # Your code here
        pass


class VQVAE(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int,
                 num_embeddings: int, embedding_dim: int):
        """
        Initialize VQ-VAE for (batch, input_channels, 4, 4) inputs.

        Args:
            input_channels: Number of input channels
            hidden_dim: Hidden layer dimension
            num_embeddings: Codebook size
            embedding_dim: Embedding dimension
        """
        super().__init__()
        torch.manual_seed(42)
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim

        # Simplified MLP encoder/decoder on the flattened 4x4 input
        self.enc1 = nn.Linear(input_channels * 16, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.dec1 = nn.Linear(embedding_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_channels * 16)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, channels, 4, 4)

        Returns:
            z_e: Encoder output (batch, 1, 1, embedding_dim)
        """
        # Your code here
        pass

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: Quantized vectors (batch, 1, 1, embedding_dim)

        Returns:
            x_recon: Reconstructed input (batch, channels, 4, 4)
        """
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass with the straight-through estimator.

        Returns:
            Dictionary with 'z_e', 'z_q', 'indices', 'x_recon'
        """
        # Your code here
        pass

    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                     z_e: torch.Tensor, z_q: torch.Tensor, beta: float = 0.25) -> dict:
        """
        Compute total VQ-VAE loss.

        Returns:
            Dictionary with 'total_loss', 'recon_loss', 'codebook_loss', 'commitment_loss'
        """
        # Your code here
        pass

    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute codebook usage statistics.

        Returns:
            usage: Count of each codebook entry used (num_embeddings,)
        """
        # Your code here
        pass
`,solution:`import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        torch.manual_seed(42)
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.1)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def quantize(self, z_e: torch.Tensor) -> tuple:
        batch, H, W, D = z_e.shape
        z_flat = z_e.reshape(-1, D)

        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z.e  -> (N, K)
        distances = ((z_flat ** 2).sum(dim=1, keepdim=True)
                     + (self.embedding ** 2).sum(dim=1)
                     - 2 * z_flat @ self.embedding.T)

        indices_flat = distances.argmin(dim=1)
        z_q = self.embedding[indices_flat].reshape(batch, H, W, D)
        indices = indices_flat.reshape(batch, H, W)
        min_distances = distances.min(dim=1)[0].reshape(batch, H, W)
        return z_q, indices, min_distances

    def compute_loss(self, z_e: torch.Tensor, z_q: torch.Tensor, beta: float = 0.25) -> dict:
        # Codebook loss moves the codes toward the (frozen) encoder output
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        # Commitment loss keeps the encoder close to the (frozen) codes
        commitment_loss = beta * F.mse_loss(z_e, z_q.detach())
        return {'codebook_loss': codebook_loss, 'commitment_loss': commitment_loss}


class VQVAE(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int,
                 num_embeddings: int, embedding_dim: int):
        super().__init__()
        torch.manual_seed(42)
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.enc1 = nn.Linear(input_channels * 16, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.dec1 = nn.Linear(embedding_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_channels * 16)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        h = torch.relu(self.enc1(x.reshape(batch, -1)))
        return self.enc2(h).reshape(batch, 1, 1, self.embedding_dim)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        batch = z_q.shape[0]
        h = torch.relu(self.dec1(z_q.reshape(batch, -1)))
        return self.dec2(h).reshape(batch, self.input_channels, 4, 4)

    def forward(self, x: torch.Tensor) -> dict:
        z_e = self.encode(x)
        z_q, indices, _ = self.vq.quantize(z_e)

        # Straight-through estimator: forward uses z_q, backward copies grads to z_e
        z_q_st = z_e + (z_q - z_e).detach()
        x_recon = self.decode(z_q_st)
        return {'z_e': z_e, 'z_q': z_q, 'indices': indices, 'x_recon': x_recon}

    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                     z_e: torch.Tensor, z_q: torch.Tensor, beta: float = 0.25) -> dict:
        recon_loss = F.mse_loss(x_recon, x)
        vq_losses = self.vq.compute_loss(z_e, z_q, beta)
        total_loss = recon_loss + vq_losses['codebook_loss'] + vq_losses['commitment_loss']
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'codebook_loss': vq_losses['codebook_loss'],
            'commitment_loss': vq_losses['commitment_loss'],
        }

    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.bincount(indices.flatten(), minlength=self.vq.num_embeddings)
`,testCases:[{id:"1",description:"Quantization shape",input:"(lambda: (vq := VectorQuantizer(512, 64), z_e := torch.randn(2, 4, 4, 64), q := vq.quantize(z_e), q[0].shape == z_e.shape)[-1])()",expected:"True",hidden:!1},{id:"2",description:"Indices valid range",input:"(lambda: (vq := VectorQuantizer(512, 64), z_e := torch.randn(2, 4, 4, 64), q := vq.quantize(z_e), bool((q[1] >= 0).all() and (q[1] < 512).all() and q[1].shape == (2, 4, 4)))[-1])()",expected:"True",hidden:!1},{id:"3",description:"Quantized from codebook, with brute-force nearest codes and distances",input:`(lambda: (
            vq := VectorQuantizer(16, 4), z_e := torch.randn(2, 3, 3, 4), q := vq.quantize(z_e),
            d := ((z_e.unsqueeze(-2) - vq.embedding.detach()) ** 2).sum(dim=-1),
            bool(torch.allclose(q[0], vq.embedding[q[1]]) and torch.equal(q[1], d.argmin(dim=-1)) and torch.allclose(q[2], d.min(dim=-1)[0], atol=1e-4))
        )[-1])()`,expected:"True",hidden:!0},{id:"4",description:"Loss values correct",input:'(lambda: (vq := VectorQuantizer(8, 2), losses := vq.compute_loss(torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), 0.25), (round(float(losses["codebook_loss"]), 4), round(float(losses["commitment_loss"]), 4)))[-1])()',expected:"(1.0, 0.25)",hidden:!0},{id:"5",description:"Straight-through: gradients reach the encoder, decoder and codebook",input:`(lambda: (
            m := VQVAE(1, 16, 8, 4), x := torch.randn(3, 1, 4, 4),
            out := m(x),
            loss := m.compute_loss(x, out["x_recon"], out["z_e"], out["z_q"]),
            loss["total_loss"].backward(),
            bool(out["x_recon"].shape == (3, 1, 4, 4) and all(p.grad is not None for p in m.parameters()) and m.vq.embedding.grad.abs().sum().item() > 0)
        )[-1])()`,expected:"True",hidden:!0},{id:"6",description:"Codebook usage counts",input:"VQVAE(1, 16, 8, 4).get_codebook_usage(torch.tensor([[0, 1], [1, 3]])).tolist()",expected:"[1, 2, 0, 1, 0, 0, 0, 0]",hidden:!0}],hints:["Flatten z_e to (N, D); distances = (z**2).sum(1, keepdim=True) + (e**2).sum(1) - 2 * z @ e.T","argmin over the codebook axis gives indices; self.embedding[indices] gives z_q","Losses: F.mse_loss(z_q, z_e.detach()) and beta * F.mse_loss(z_e, z_q.detach())","Straight-through: decode z_e + (z_q - z_e).detach()","Usage: torch.bincount(indices.flatten(), minlength=K)"]},"e2e-diffusion":{note:"DiffusionModel is an nn.Module that stores its schedule with register_buffer. Use torch.linspace/cumprod and torch.randn_like. Tests use a small T so sampling stays fast.",starterCode:`import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, T: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        """
        Initialize diffusion model.

        Args:
            T: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        super().__init__()
        self.T = T

        # Compute noise schedule and store it with self.register_buffer(name, tensor)
        # Your code here

    def get_noise_schedule(self) -> dict:
        """Return computed noise schedule parameters ('betas', 'alphas', 'alpha_bars')."""
        # Your code here
        pass

    def forward_diffusion(self, x_0: torch.Tensor, t: int) -> tuple:
        """
        Add noise to data at timestep t.

        Args:
            x_0: Original data (batch, ...)
            t: Timestep

        Returns:
            x_t: Noisy data
            noise: The noise that was added
        """
        torch.manual_seed(42)
        # Your code here
        pass

    def reverse_step(self, x_t: torch.Tensor, t: int, predicted_noise: torch.Tensor) -> torch.Tensor:
        """
        Single reverse diffusion step (adds sigma_t * z for t > 0, none at t = 0).

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            predicted_noise: Noise predicted by model

        Returns:
            x_{t-1}: Less noisy data
        """
        # Your code here
        pass

    def compute_loss(self, true_noise: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between true and predicted noise (scalar tensor)."""
        # Your code here
        pass

    def sample(self, shape: tuple, noise_predictor) -> torch.Tensor:
        """
        Generate samples using reverse diffusion.

        Args:
            shape: Shape of samples to generate
            noise_predictor: Function that predicts noise given (x_t, t)

        Returns:
            Generated samples
        """
        # Your code here
        pass
`,solution:`import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, T: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        super().__init__()
        self.T = T

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Buffers: part of the module's state, but not learnable parameters
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

    def get_noise_schedule(self) -> dict:
        return {'betas': self.betas, 'alphas': self.alphas, 'alpha_bars': self.alpha_bars}

    def forward_diffusion(self, x_0: torch.Tensor, t: int) -> tuple:
        torch.manual_seed(42)
        noise = torch.randn_like(x_0)
        alpha_bar = self.alpha_bars[t]
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def reverse_step(self, x_t: torch.Tensor, t: int, predicted_noise: torch.Tensor) -> torch.Tensor:
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]

        # Mean of p(x_{t-1} | x_t)
        mean = (x_t - beta / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)

        if t > 0:
            return mean + torch.sqrt(beta) * torch.randn_like(x_t)
        return mean

    def compute_loss(self, true_noise: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predicted_noise, true_noise)

    @torch.no_grad()
    def sample(self, shape: tuple, noise_predictor) -> torch.Tensor:
        x = torch.randn(*shape)
        for t in reversed(range(self.T)):
            x = self.reverse_step(x, t, noise_predictor(x, t))
        return x
`,testCases:[{id:"1",description:"Alpha bar decreases",input:'(lambda: (dm := DiffusionModel(1000), sched := dm.get_noise_schedule(), bool(sched["alpha_bars"][-1] < sched["alpha_bars"][0]))[-1])()',expected:"True",hidden:!1},{id:"2",description:"Forward noise shape",input:"(lambda: (dm := DiffusionModel(1000), x_0 := torch.randn(4, 32), fd := dm.forward_diffusion(x_0, 500), fd[0].shape == x_0.shape)[-1])()",expected:"True",hidden:!1},{id:"3",description:"Schedule values for T=3",input:`(lambda: (
            s := DiffusionModel(3, 0.1, 0.3).get_noise_schedule(),
            bool(torch.allclose(s["betas"], torch.tensor([0.1, 0.2, 0.3])) and torch.allclose(s["alphas"], torch.tensor([0.9, 0.8, 0.7])) and torch.allclose(s["alpha_bars"], torch.tensor([0.9, 0.72, 0.504])))
        )[-1])()`,expected:"True",hidden:!1},{id:"4",description:"Forward process follows x_t = sqrt(ab) x_0 + sqrt(1 - ab) noise",input:`(lambda: (
            dm := DiffusionModel(100), x0 := torch.randn(3, 5), r := dm.forward_diffusion(x0, 40), ab := dm.alpha_bars[40],
            bool(torch.allclose(r[0], ab.sqrt() * x0 + (1 - ab).sqrt() * r[1], atol=1e-6))
        )[-1])()`,expected:"True",hidden:!0},{id:"5",description:"Reverse step at t=0 is the deterministic posterior mean",input:`(lambda: (
            dm := DiffusionModel(10, 0.1, 0.5), x := torch.tensor([[1.0, -2.0]]), eps := torch.tensor([[0.5, 0.5]]),
            ref := (x - 0.1 / (1 - 0.9) ** 0.5 * eps) / 0.9 ** 0.5,
            bool(torch.allclose(dm.reverse_step(x, 0, eps), ref, atol=1e-5))
        )[-1])()`,expected:"True",hidden:!0},{id:"6",description:"Noise prediction loss is the MSE",input:"round(float(DiffusionModel(10).compute_loss(torch.zeros(2, 2), torch.tensor([[1.0, -1.0], [2.0, 0.0]]))), 4)",expected:"1.5",hidden:!0},{id:"7",description:"Sampling runs T reverse steps and keeps the shape; schedule lives in buffers",input:`(lambda: (
            dm := DiffusionModel(10), calls := [],
            s := dm.sample((3, 2), lambda x, t: (calls.append(t), torch.zeros_like(x))[1]),
            bool(s.shape == (3, 2) and calls == list(range(9, -1, -1)) and isinstance(dm, torch.nn.Module) and len(list(dm.buffers())) >= 3)
        )[-1])()`,expected:"True",hidden:!0}],hints:["betas = torch.linspace(beta_start, beta_end, T); alpha_bars = torch.cumprod(1 - betas, dim=0)","self.register_buffer('betas', betas) makes the schedule part of the module state","x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise","Reverse mean: (x_t - beta / sqrt(1 - alpha_bar) * eps) / sqrt(alpha); add sqrt(beta) * z only for t > 0"]},"e2e-cnn":{note:"Write conv2d and maxpool2d with tensor ops: don't use nn.Conv2d, F.conv2d, nn.MaxPool2d, F.max_pool2d or unfold (hidden tests compare against them). CNN is an nn.Module; nn.Linear is fine for the fully connected layers.",starterCode:`import torch
import torch.nn as nn


def conv2d(x: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    2D convolution (no padding, stride=1).

    Args:
        x: Input (batch, in_channels, height, width)
        kernel: Weights (out_channels, in_channels, kH, kW)
        bias: Bias (out_channels,)

    Returns:
        output: (batch, out_channels, out_height, out_width)
    """
    # Your code here
    pass


def maxpool2d(x: torch.Tensor, pool_size: int = 2) -> torch.Tensor:
    """
    2D max pooling (window = stride = pool_size).

    Args:
        x: Input (batch, channels, height, width)
        pool_size: Size of pooling window

    Returns:
        output: Pooled tensor
    """
    # Your code here
    pass


def flatten(x: torch.Tensor) -> torch.Tensor:
    """Flatten all dimensions except batch."""
    # Your code here
    pass


class CNN(nn.Module):
    def __init__(self, input_shape: tuple, num_classes: int):
        """
        Initialize CNN.

        Args:
            input_shape: (channels, height, width)
            num_classes: Number of output classes
        """
        super().__init__()
        torch.manual_seed(42)
        in_channels, H, W = input_shape

        # Conv layer 1: in_channels -> 32
        self.conv1_w = nn.Parameter(torch.randn(32, in_channels, 3, 3) * 0.1)
        self.conv1_b = nn.Parameter(torch.zeros(32))

        # Conv layer 2: 32 -> 64
        self.conv2_w = nn.Parameter(torch.randn(64, 32, 3, 3) * 0.1)
        self.conv2_b = nn.Parameter(torch.zeros(64))

        # Calculate flattened size after convolutions and pooling
        # Your code to compute this

        # Fully connected layers (nn.Linear)
        # Your code here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.

        Args:
            x: Input images (batch, channels, height, width)

        Returns:
            logits: (batch, num_classes)
        """
        # Your code here
        pass


def test_cnn():
    """Test CNN with sample data; return True if the output shape is right."""
    # Your code here
    pass
`,solution:`import torch
import torch.nn as nn


def conv2d(x: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    batch, in_ch, H, W = x.shape
    out_ch, _, kH, kW = kernel.shape
    out_H, out_W = H - kH + 1, W - kW + 1

    # For each kernel tap (a, b): contract the shifted input over input channels
    output = torch.zeros(batch, out_ch, out_H, out_W)
    for a in range(kH):
        for b in range(kW):
            patch = x[:, :, a:a + out_H, b:b + out_W]               # (batch, in_ch, out_H, out_W)
            output = output + torch.einsum('bchw,oc->bohw', patch, kernel[:, :, a, b])

    return output + bias.view(1, -1, 1, 1)


def maxpool2d(x: torch.Tensor, pool_size: int = 2) -> torch.Tensor:
    batch, channels, H, W = x.shape
    p = pool_size
    out_H, out_W = H // p, W // p

    # Crop, split each spatial axis into (blocks, p) and take the max inside each block
    x = x[:, :, :out_H * p, :out_W * p].reshape(batch, channels, out_H, p, out_W, p)
    return x.max(dim=5)[0].max(dim=3)[0]


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


class CNN(nn.Module):
    def __init__(self, input_shape: tuple, num_classes: int):
        super().__init__()
        torch.manual_seed(42)
        in_channels, H, W = input_shape

        self.conv1_w = nn.Parameter(torch.randn(32, in_channels, 3, 3) * 0.1)
        self.conv1_b = nn.Parameter(torch.zeros(32))
        self.conv2_w = nn.Parameter(torch.randn(64, 32, 3, 3) * 0.1)
        self.conv2_b = nn.Parameter(torch.zeros(64))

        # conv 3x3 (-2) -> pool 2 (//2) -> conv 3x3 (-2) -> pool 2 (//2)
        h2 = ((H - 2) // 2 - 2) // 2
        w2 = ((W - 2) // 2 - 2) // 2
        flat_size = 64 * h2 * w2

        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = maxpool2d(torch.relu(conv2d(x, self.conv1_w, self.conv1_b)))
        x = maxpool2d(torch.relu(conv2d(x, self.conv2_w, self.conv2_b)))
        x = flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_cnn():
    cnn = CNN(input_shape=(1, 28, 28), num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    logits = cnn(x)
    return tuple(logits.shape) == (4, 10)
`,testCases:[{id:"1",description:"Conv output shape",input:"conv2d(torch.randn(2, 1, 8, 8), torch.randn(32, 1, 3, 3), torch.zeros(32)).shape",expected:"(2, 32, 6, 6)",hidden:!1},{id:"2",description:"Pool output shape",input:"maxpool2d(torch.randn(2, 32, 8, 8), pool_size=2).shape",expected:"(2, 32, 4, 4)",hidden:!1},{id:"3",description:"Conv values with bias (hand-computed)",input:"conv2d(torch.ones(1, 2, 3, 3), torch.ones(1, 2, 2, 2), torch.tensor([1.0])).tolist()",expected:"[[[[9.0, 9.0], [9.0, 9.0]]]]",hidden:!1},{id:"4",description:"conv2d matches F.conv2d on random multi-channel input",input:`(lambda: (
            x := torch.randn(2, 3, 7, 6), k := torch.randn(4, 3, 3, 2), b := torch.randn(4),
            bool(torch.allclose(conv2d(x, k, b), torch.nn.functional.conv2d(x, k, b), atol=1e-4))
        )[-1])()`,expected:"True",hidden:!0},{id:"5",description:"maxpool2d matches F.max_pool2d",input:`(lambda: (
            x := torch.randn(2, 3, 9, 8),
            bool(torch.allclose(maxpool2d(x, 2), torch.nn.functional.max_pool2d(x, 2)) and torch.allclose(maxpool2d(x, 3), torch.nn.functional.max_pool2d(x, 3)))
        )[-1])()`,expected:"True",hidden:!0},{id:"6",description:"CNN forward shape, nn.Module, and gradients reach every parameter",input:`(lambda: (
            m := CNN((1, 12, 12), 3), out := m(torch.randn(2, 1, 12, 12)),
            out.sum().backward(),
            bool(out.shape == (2, 3) and isinstance(m, torch.nn.Module) and all(p.grad is not None for p in m.parameters()))
        )[-1])()`,expected:"True",hidden:!0},{id:"7",description:"test_cnn runs on 28x28 inputs",input:"test_cnn()",expected:"True",hidden:!0}],hints:["Conv output size: (input - kernel + 1) for stride=1, no padding",'Loop over kernel taps: output += einsum("bchw,oc->bohw", x[:, :, a:a+out_H, b:b+out_W], kernel[:, :, a, b])',"Pool: reshape to (B, C, out_H, p, out_W, p) and max over the two p axes","Flatten: x.reshape(x.shape[0], -1); FC layers: nn.Linear"]}},hp={...Ix,...Dx,...Mx,...qx,...Ox,...Wx,...Vx,...Bx,...Ux,...Hx,...Qx,...Xx,...Kx,...Yx,...$x,...Gx,...Zx,...Jx},eb=[{id:"numpy-array-creation",title:"Array Creation Methods",section:"numpy-fundamentals",difficulty:"easy",description:`
## NumPy Array Creation

Implement a function that creates various NumPy arrays using different methods.

### Requirements
Create and return a dictionary with:
- \`zeros\`: 3x4 array of zeros
- \`ones\`: 2x3 array of ones
- \`arange\`: array from 0 to 9
- \`linspace\`: 5 evenly spaced values from 0 to 1
- \`eye\`: 4x4 identity matrix

### Function Signature
\`\`\`python
def create_arrays() -> dict:
\`\`\`
    `,examples:[{input:"create_arrays()",output:"{'zeros': array([[0,0,0,0],...]), ...}",explanation:"Dictionary with 5 different array types"}],starterCode:`import numpy as np

def create_arrays() -> dict:
    """
    Create various NumPy arrays.

    Returns:
        Dictionary with keys: zeros, ones, arange, linspace, eye
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Zeros shape correct",input:'create_arrays()["zeros"].shape',expected:"(3, 4)",hidden:!1},{id:"2",description:"Linspace length correct",input:'len(create_arrays()["linspace"])',expected:"5",hidden:!1},{id:"3",description:"Eye is identity",input:'bool(np.allclose(create_arrays()["eye"], np.eye(4)))',expected:"True",hidden:!0}],hints:["Use np.zeros((3, 4)) for zero array","np.linspace(start, stop, num) creates evenly spaced values","np.eye(n) creates an identity matrix"],solution:`import numpy as np

def create_arrays() -> dict:
    return {
        'zeros': np.zeros((3, 4)),
        'ones': np.ones((2, 3)),
        'arange': np.arange(10),
        'linspace': np.linspace(0, 1, 5),
        'eye': np.eye(4)
    }
`},{id:"numpy-indexing",title:"Advanced Indexing",section:"numpy-fundamentals",difficulty:"medium",description:`
## NumPy Advanced Indexing

Implement various indexing operations on a 2D array.

### Given
\`\`\`python
arr = np.arange(20).reshape(4, 5)
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19]])
\`\`\`

### Requirements
Return a dictionary with:
- \`row_1\`: Second row (index 1)
- \`col_2\`: Third column (index 2)
- \`subarray\`: Rows 1-2, columns 2-4
- \`diagonal\`: Main diagonal
- \`reversed\`: Array with rows reversed

### Function Signature
\`\`\`python
def advanced_indexing(arr: np.ndarray) -> dict:
\`\`\`
    `,examples:[{input:"arr = np.arange(20).reshape(4, 5)",output:"{'row_1': [5,6,7,8,9], ...}",explanation:"Various slicing and indexing operations"}],starterCode:`import numpy as np

def advanced_indexing(arr: np.ndarray) -> dict:
    """
    Perform advanced indexing operations.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with indexed results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Row extraction",input:'advanced_indexing(np.arange(20).reshape(4, 5))["row_1"].tolist()',expected:"[5, 6, 7, 8, 9]",hidden:!1},{id:"2",description:"Subarray correct",input:'advanced_indexing(np.arange(20).reshape(4, 5))["subarray"].tolist()',expected:"[[7, 8, 9], [12, 13, 14]]",hidden:!1},{id:"3",description:"Diagonal and reversed rows on a 4x5 array",input:'(lambda r: (r["diagonal"].tolist(), r["reversed"].tolist()[0], r["col_2"].tolist()))(advanced_indexing(np.arange(20).reshape(4, 5) * 2))',expected:"([0, 12, 24, 36], [30, 32, 34, 36, 38], [4, 14, 24, 34])",hidden:!0}],hints:["arr[1] or arr[1, :] gets row 1","arr[:, 2] gets column 2","arr[1:3, 2:5] gets a subarray","np.diag(arr) extracts diagonal","arr[::-1] reverses rows"],solution:`import numpy as np

def advanced_indexing(arr: np.ndarray) -> dict:
    return {
        'row_1': arr[1],
        'col_2': arr[:, 2],
        'subarray': arr[1:3, 2:5],
        'diagonal': np.diag(arr),
        'reversed': arr[::-1]
    }
`},{id:"numpy-broadcasting",title:"Broadcasting Operations",section:"numpy-fundamentals",difficulty:"medium",description:`
## NumPy Broadcasting

Implement operations that leverage NumPy's broadcasting rules.

### Task
Given a 2D array of shape (3, 4):
1. Subtract the row means from each row (center rows)
2. Subtract the column means from each column (center columns)
3. Add a 1D bias vector of shape (4,) to each row
4. Multiply by a column vector of shape (3, 1)

### Broadcasting Rules
- Arrays are compatible when dimensions are equal or one of them is 1
- Operations proceed element-wise on matching dimensions

### Function Signature
\`\`\`python
def broadcasting_ops(arr: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> dict:
\`\`\`
    `,examples:[{input:"arr (3,4), bias (4,), scale (3,1)",output:"{'row_centered': ..., 'col_centered': ..., 'biased': ..., 'scaled': ...}",explanation:"Various broadcasting operations"}],starterCode:`import numpy as np

def broadcasting_ops(arr: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> dict:
    """
    Perform broadcasting operations.

    Args:
        arr: 2D array of shape (3, 4)
        bias: 1D array of shape (4,)
        scale: 2D array of shape (3, 1)

    Returns:
        Dictionary with results of broadcasting operations
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Row centering",input:'bool(np.allclose(broadcasting_ops(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]), np.zeros(4), np.ones((3,1)))["row_centered"].mean(axis=1), 0))',expected:"True",hidden:!1},{id:"2",description:"Bias addition shape",input:'broadcasting_ops(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]), np.array([1,1,1,1]), np.ones((3,1)))["biased"].shape',expected:"(3, 4)",hidden:!1},{id:"3",description:"Scaled and column-centered values",input:'(lambda r: (r["scaled"].tolist(), r["col_centered"].tolist()))(broadcasting_ops(np.array([[1.0, 2.0], [3.0, 6.0]]), np.array([0.5, -0.5]), np.array([[2.0], [10.0]])))',expected:"([[2.0, 4.0], [30.0, 60.0]], [[-1.0, -2.0], [1.0, 2.0]])",hidden:!0}],hints:["Row means: arr.mean(axis=1, keepdims=True)","Column means: arr.mean(axis=0, keepdims=True) or arr.mean(axis=0)","Bias shape (4,) broadcasts with arr shape (3, 4)","Scale shape (3, 1) broadcasts with arr shape (3, 4)"],solution:`import numpy as np

def broadcasting_ops(arr: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> dict:
    # Row centered: subtract mean of each row
    row_means = arr.mean(axis=1, keepdims=True)  # (3, 1)
    row_centered = arr - row_means

    # Column centered: subtract mean of each column
    col_means = arr.mean(axis=0)  # (4,)
    col_centered = arr - col_means

    # Add bias to each row
    biased = arr + bias  # bias (4,) broadcasts to (3, 4)

    # Scale each row
    scaled = arr * scale  # scale (3, 1) broadcasts to (3, 4)

    return {
        'row_centered': row_centered,
        'col_centered': col_centered,
        'biased': biased,
        'scaled': scaled
    }
`},{id:"numpy-aggregations",title:"Aggregation Functions",section:"numpy-fundamentals",difficulty:"easy",description:"\n## NumPy Aggregations\n\nImplement common aggregation operations along different axes.\n\n### Task\nGiven a 2D array, compute:\n- Global statistics (mean, std, min, max, sum)\n- Row-wise statistics (along axis=1)\n- Column-wise statistics (along axis=0)\n- Argmax and argmin (indices of max/min values)\n\n### Expected Return Format\nReturn a dictionary with these keys:\n- **Global**: `'global_mean'`, `'global_std'`, `'global_min'`, `'global_max'`, `'global_sum'`\n- **Row-wise**: `'row_mean'`, `'row_sum'`\n- **Column-wise**: `'col_mean'`, `'col_sum'`\n- **Indices**: `'argmax'` (2D index tuple), `'argmin'` (2D index tuple)\n    ",examples:[{input:"np.array([[1, 2, 3], [4, 5, 6]])",output:"{'global_mean': 3.5, 'global_sum': 21, 'row_sum': [6, 15], 'col_mean': [2.5, 3.5, 4.5], 'argmax': (1, 2), ...}",explanation:"Global, row-wise, and column-wise aggregations"}],starterCode:`import numpy as np

def compute_aggregations(arr: np.ndarray) -> dict:
    """
    Compute aggregation statistics.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with aggregation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Global mean",input:'compute_aggregations(np.array([[1, 2, 3], [4, 5, 6]]))["global_mean"]',expected:"3.5",hidden:!1},{id:"2",description:"Row sums",input:'compute_aggregations(np.array([[1, 2, 3], [4, 5, 6]]))["row_sum"].tolist()',expected:"[6, 15]",hidden:!1},{id:"3",description:"Stats on a 2x3 array with negatives",input:'(lambda r: (round(float(r["global_std"]), 4), r["col_mean"].tolist(), tuple(int(i) for i in r["argmax"]), tuple(int(i) for i in r["argmin"])))(compute_aggregations(np.array([[3.0, -1.0, 2.0], [0.5, 4.0, -2.0]])))',expected:"(2.1295, [1.75, 1.5, 0.0], (1, 1), (1, 2))",hidden:!0}],hints:["arr.mean() computes global mean","arr.mean(axis=0) computes column means","arr.mean(axis=1) computes row means","np.argmax(arr) gives index of max in flattened array"],solution:`import numpy as np

def compute_aggregations(arr: np.ndarray) -> dict:
    return {
        'global_mean': arr.mean(),
        'global_std': arr.std(),
        'global_min': arr.min(),
        'global_max': arr.max(),
        'global_sum': arr.sum(),
        'row_mean': arr.mean(axis=1),
        'row_sum': arr.sum(axis=1),
        'col_mean': arr.mean(axis=0),
        'col_sum': arr.sum(axis=0),
        'argmax': np.unravel_index(arr.argmax(), arr.shape),
        'argmin': np.unravel_index(arr.argmin(), arr.shape),
    }
`},{id:"numpy-reshape-transpose",title:"Reshape and Transpose",section:"numpy-fundamentals",difficulty:"medium",description:`
## Reshape and Transpose Operations

Master array shape manipulation - essential for ML data pipelines.

### Task
Given a 1D array of 24 elements:
1. Reshape to (4, 6)
2. Reshape to (2, 3, 4)
3. Transpose the 2D version
4. Reverse the axes of the 3D version (transpose(2, 1, 0))
5. Flatten back to 1D

### Important Concepts
- \`reshape\` changes shape without changing data order
- \`transpose\` swaps axes
- \`-1\` in reshape means "infer this dimension"

### Expected Return Format
Return a dictionary with these keys:
- \`'arr_2d'\`: Reshaped to (4, 6)
- \`'arr_3d'\`: Reshaped to (2, 3, 4)
- \`'arr_2d_transposed'\`: Transpose of arr_2d, shape (6, 4)
- \`'arr_3d_swapped'\`: arr_3d with axes swapped (2,1,0), shape (4, 3, 2)
- \`'arr_flat'\`: Flattened back to 1D, shape (24,)
    `,examples:[{input:"np.arange(24)",output:"{'arr_2d': shape (4,6), 'arr_3d': shape (2,3,4), 'arr_2d_transposed': shape (6,4), ...}",explanation:"Various reshape and transpose operations"}],starterCode:`import numpy as np

def reshape_transpose(arr: np.ndarray) -> dict:
    """
    Perform reshape and transpose operations.

    Args:
        arr: 1D array of 24 elements

    Returns:
        Dictionary with reshaped arrays
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"2D reshape shape",input:'reshape_transpose(np.arange(24))["arr_2d"].shape',expected:"(4, 6)",hidden:!1},{id:"2",description:"3D reshape shape",input:'reshape_transpose(np.arange(24))["arr_3d"].shape',expected:"(2, 3, 4)",hidden:!1},{id:"3",description:"Values, not just shapes",input:'(lambda r: (r["arr_2d_transposed"][0].tolist(), r["arr_3d_swapped"].shape, r["arr_flat"][:5].tolist()))(reshape_transpose(np.arange(24)))',expected:"([0, 6, 12, 18], (4, 3, 2), [0, 1, 2, 3, 4])",hidden:!0}],hints:["arr.reshape(4, 6) or arr.reshape(4, -1)","arr.reshape(2, 3, 4) for 3D",".T or .transpose() for transpose",".flatten() or .ravel() for 1D"],solution:`import numpy as np

def reshape_transpose(arr: np.ndarray) -> dict:
    # 2D reshape
    arr_2d = arr.reshape(4, 6)

    # 3D reshape
    arr_3d = arr.reshape(2, 3, 4)

    # Transpose 2D
    arr_2d_T = arr_2d.T

    # Swap axes on 3D (swap axis 0 and 2)
    arr_3d_swapped = arr_3d.transpose(2, 1, 0)

    # Flatten
    arr_flat = arr_2d.flatten()

    return {
        'arr_2d': arr_2d,
        'arr_3d': arr_3d,
        'arr_2d_transposed': arr_2d_T,
        'arr_3d_swapped': arr_3d_swapped,
        'arr_flat': arr_flat
    }
`},{id:"numpy-array-sum",title:"Array Sum",section:"numpy-fundamentals",difficulty:"easy",description:`
## Array Sum

Given a NumPy array of numbers, implement a function that returns the sum of all elements.

### Function Signature
\`\`\`python
def array_sum(arr: np.ndarray) -> float:
\`\`\`

### Constraints
- Array length: 1 <= n <= 1000
- Element values: -10^6 <= arr[i] <= 10^6
    `,examples:[{input:"np.array([1, 2, 3, 4, 5])",output:"15",explanation:"1 + 2 + 3 + 4 + 5 = 15"},{input:"np.array([-1, 0, 1])",output:"0",explanation:"-1 + 0 + 1 = 0"}],starterCode:`import numpy as np

def array_sum(arr: np.ndarray) -> float:
    """
    Calculate the sum of all elements in a NumPy array.

    Args:
        arr: A NumPy array of numbers

    Returns:
        The sum of all elements
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic positive numbers",input:"[1, 2, 3, 4, 5]",expected:"15",hidden:!1},{id:"2",description:"With negatives",input:"[-1, 0, 1]",expected:"0",hidden:!1},{id:"3",description:"Single element",input:"[42]",expected:"42",hidden:!1},{id:"4",description:"Larger array",input:"[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",expected:"55",hidden:!0}],hints:["NumPy arrays have built-in methods for common operations.","Try using np.sum() or the .sum() method on the array."],solution:`import numpy as np

def array_sum(arr: np.ndarray) -> float:
    """
    Calculate the sum of all elements in a NumPy array.
    """
    return np.sum(arr)
`},{id:"numpy-matrix-multiply",title:"Matrix Multiplication",section:"numpy-fundamentals",difficulty:"medium",description:`
## Matrix Multiplication

Implement matrix multiplication between two 2D NumPy arrays.

### Function Signature
\`\`\`python
def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- A has shape (m, n)
- B has shape (n, p)
- Result has shape (m, p)
- 1 <= m, n, p <= 100

### Note
You should use NumPy's built-in functions, not manual loops.
    `,examples:[{input:"A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]",output:"[[19, 22], [43, 50]]",explanation:"Standard matrix multiplication"}],starterCode:`import numpy as np

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices A and B.

    Args:
        A: First matrix of shape (m, n)
        B: Second matrix of shape (n, p)

    Returns:
        Result matrix of shape (m, p)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"2x2 matrices",input:"([[1, 2], [3, 4]], [[5, 6], [7, 8]])",expected:"[[19, 22], [43, 50]]",hidden:!1},{id:"2",description:"Identity multiplication",input:"([[1, 0], [0, 1]], [[5, 6], [7, 8]])",expected:"[[5, 6], [7, 8]]",hidden:!1},{id:"3",description:"Different dimensions",input:"([[1, 2, 3]], [[1], [2], [3]])",expected:"[[14]]",hidden:!0}],hints:["NumPy provides several ways to do matrix multiplication.","Try np.dot(), np.matmul(), or the @ operator."],solution:`import numpy as np

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices A and B.
    """
    return np.dot(A, B)
    # Alternative: return A @ B
    # Alternative: return np.matmul(A, B)
`},{id:"numpy-broadcast-add",title:"Broadcasting Addition",section:"numpy-fundamentals",difficulty:"medium",description:`
## Broadcasting Addition

Given a 2D matrix and a 1D vector, add the vector to each row of the matrix using broadcasting.

### Function Signature
\`\`\`python
def broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- Matrix has shape (m, n)
- Vector has shape (n,)
- 1 <= m, n <= 100

### Note
Broadcasting is a powerful NumPy feature that allows operations between arrays of different shapes.
    `,examples:[{input:"matrix = [[1, 2, 3], [4, 5, 6]], vector = [10, 20, 30]",output:"[[11, 22, 33], [14, 25, 36]]",explanation:"Vector is added to each row"}],starterCode:`import numpy as np

def broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Add a vector to each row of a matrix using broadcasting.

    Args:
        matrix: 2D array of shape (m, n)
        vector: 1D array of shape (n,)

    Returns:
        Result matrix of shape (m, n)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic broadcasting",input:"([[1, 2, 3], [4, 5, 6]], [10, 20, 30])",expected:"[[11, 22, 33], [14, 25, 36]]",hidden:!1},{id:"2",description:"With zeros",input:"([[1, 2], [3, 4]], [0, 0])",expected:"[[1, 2], [3, 4]]",hidden:!1},{id:"3",description:"Negative values",input:"([[1, 2], [3, 4]], [-1, -2])",expected:"[[0, 0], [2, 2]]",hidden:!0}],hints:["NumPy automatically broadcasts the vector across rows.","Simply use the + operator - NumPy handles the rest!"],solution:`import numpy as np

def broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Add a vector to each row of a matrix using broadcasting.
    """
    return matrix + vector
`}],tb=[{id:"einsum-basics",title:"Einsum Fundamentals",section:"einsum",difficulty:"easy",description:`
## Einsum Fundamentals

Einstein summation notation provides a powerful way to express array operations.

### Syntax
\`\`\`python
np.einsum('subscripts', operands)
\`\`\`

### Basic Operations
- \`'i->'\`: Sum all elements (reduce)
- \`'ij->i'\`: Sum along columns (row sums)
- \`'ij->j'\`: Sum along rows (column sums)
- \`'ii->i'\`: Extract diagonal
- \`'ij->ji'\`: Transpose

### Task
Implement basic einsum operations on a 2D array.

### Expected Return Format
Return a dictionary with these keys:
- \`'sum_all'\`: Sum of all elements
- \`'row_sum'\`: Sum along each row
- \`'col_sum'\`: Sum along each column
- \`'transpose'\`: Transposed array
- \`'diagonal'\`: Diagonal elements (for square matrices, else None)
    `,examples:[{input:"np.array([[1,2],[3,4]])",output:"{'sum_all': 10, 'row_sum': [3,7], 'col_sum': [4,6], 'transpose': [[1,3],[2,4]], 'diagonal': [1,4]}",explanation:"Basic einsum operations on a 2x2 matrix"}],starterCode:`import numpy as np

def einsum_basics(arr: np.ndarray) -> dict:
    """
    Perform basic einsum operations.

    Args:
        arr: 2D input array

    Returns:
        Dictionary with einsum results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Sum all elements",input:'einsum_basics(np.array([[1, 2], [3, 4]]))["sum_all"]',expected:"10",hidden:!1},{id:"2",description:"Transpose",input:'bool(np.array_equal(einsum_basics(np.array([[1, 2], [3, 4]]))["transpose"], np.array([[1, 3], [2, 4]])))',expected:"True",hidden:!1},{id:"3",description:"Sums and diagonal of a 3x3 matrix",input:'(lambda r: (int(r["sum_all"]), r["row_sum"].tolist(), r["col_sum"].tolist(), r["diagonal"].tolist()))(einsum_basics(np.arange(9).reshape(3, 3)))',expected:"(36, [3, 12, 21], [9, 12, 15], [0, 4, 8])",hidden:!0}],hints:["'ij->' sums all elements","'ij->i' sums each row","'ij->ji' transposes","'ii->i' extracts diagonal (for square matrices)"],solution:`import numpy as np

def einsum_basics(arr: np.ndarray) -> dict:
    return {
        'sum_all': np.einsum('ij->', arr),
        'row_sum': np.einsum('ij->i', arr),
        'col_sum': np.einsum('ij->j', arr),
        'transpose': np.einsum('ij->ji', arr),
        'diagonal': np.einsum('ii->i', arr) if arr.shape[0] == arr.shape[1] else None
    }
`},{id:"einsum-matrix-ops",title:"Matrix Operations with Einsum",section:"einsum",difficulty:"medium",description:`
## Matrix Operations with Einsum

Einsum can express matrix multiplication and more complex operations.

### Key Operations
- \`'ik,kj->ij'\`: Matrix multiplication (A @ B)
- \`'ij,ij->ij'\`: Element-wise multiplication (Hadamard)
- \`'ij,ij->'\`: Frobenius inner product (sum of element-wise product)
- \`'ij,j->i'\`: Matrix-vector multiplication
- \`'i,j->ij'\`: Outer product

### Task
Implement matrix operations using einsum.

### Expected Return Format
Return a dictionary with these keys:
- \`'matmul'\`: Matrix multiplication A @ B
- \`'matvec'\`: Matrix-vector multiplication A @ v
- \`'outer_product'\`: Outer product of v with itself
- \`'hadamard'\`: Element-wise A * A
- \`'frobenius'\`: Sum of A * A (Frobenius inner product)
- \`'trace'\`: Trace of A (if square, else None)
    `,examples:[{input:"A=[[1,2],[3,4]], B=[[5,6],[7,8]], v=[1,2]",output:"{'matmul': [[19,22],[43,50]], 'matvec': [5,11], 'outer_product': [[1,2],[2,4]], 'hadamard': [[1,4],[9,16]], 'frobenius': 30, 'trace': 5}",explanation:"Matrix operations via einsum"}],starterCode:`import numpy as np

def einsum_matrix_ops(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> dict:
    """
    Perform matrix operations using einsum.

    Args:
        A: Matrix of shape (m, k)
        B: Matrix of shape (k, n)
        v: Vector of shape (k,)

    Returns:
        Dictionary with operation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Matrix multiplication",input:'bool(np.array_equal(einsum_matrix_ops(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]), np.array([1,2]))["matmul"], np.array([[19,22],[43,50]])))',expected:"True",hidden:!1},{id:"2",description:"Outer product",input:'bool(np.array_equal(einsum_matrix_ops(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]), np.array([1,2]))["outer_product"], np.array([[1,2],[2,4]])))',expected:"True",hidden:!1},{id:"3",description:"All ops match NumPy on 3x3 inputs",input:'(lambda A, B, v: (lambda r: bool(np.allclose(r["matmul"], A @ B) and np.allclose(r["matvec"], A @ v) and np.allclose(r["outer_product"], np.outer(v, v)) and np.allclose(r["hadamard"], A * A) and np.isclose(r["trace"], np.trace(A)) and np.isclose(r["frobenius"], np.sum(A * A))))(einsum_matrix_ops(A, B, v)))(np.random.randn(3, 3), np.random.randn(3, 3), np.random.randn(3))',expected:"True",hidden:!0}],hints:["'ik,kj->ij' contracts over k (matrix multiply)","'ij,j->i' contracts vector with columns","'i,j->ij' creates outer product (no contraction)"],solution:`import numpy as np

def einsum_matrix_ops(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> dict:
    return {
        'matmul': np.einsum('ik,kj->ij', A, B),
        'matvec': np.einsum('ij,j->i', A, v),
        'outer_product': np.einsum('i,j->ij', v, v),
        'hadamard': np.einsum('ij,ij->ij', A, A),  # A * A
        'frobenius': np.einsum('ij,ij->', A, A),   # sum(A * A)
        'trace': np.einsum('ii->', A) if A.shape[0] == A.shape[1] else None
    }
`},{id:"einsum-batch-ops",title:"Batch Operations with Einsum",section:"einsum",difficulty:"medium",description:`
## Batch Operations with Einsum

Einsum shines for batch operations - multiple matrices at once.

### Batch Matrix Operations
- \`'bij,bjk->bik'\`: Batch matrix multiplication
- \`'bij,bj->bi'\`: Batch matrix-vector multiplication
- \`'bij->bji'\`: Batch transpose

### Attention-style Operations
- \`'bqd,bkd->bqk'\`: Query-Key dot products (attention scores)
- \`'bqk,bkv->bqv'\`: Weighted sum of values

### Task
Implement batch operations common in deep learning.

### Expected Return Format
Return a dictionary with these keys:
- \`'scores'\`: Raw attention scores Q @ K.T per batch, shape (batch, seq_q, seq_k)
- \`'attention_weights'\`: Softmax of scaled scores, shape (batch, seq_q, seq_k)
- \`'output'\`: Weighted sum of values, shape (batch, seq_q, dim_v)
    `,examples:[{input:"Q (2,4,8), K (2,6,8), V (2,6,16)",output:"{'scores': shape (2,4,6), 'attention_weights': shape (2,4,6), 'output': shape (2,4,16)}",explanation:"Batch attention: scores from Q@K.T, then weighted sum of V"}],starterCode:`import numpy as np

def einsum_batch_ops(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> dict:
    """
    Perform batch operations using einsum.

    Args:
        Q: Queries of shape (batch, seq_q, dim)
        K: Keys of shape (batch, seq_k, dim)
        V: Values of shape (batch, seq_k, dim_v)

    Returns:
        Dictionary with batch operation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Attention scores shape",input:'einsum_batch_ops(np.ones((2, 4, 8)), np.ones((2, 6, 8)), np.ones((2, 6, 16)))["scores"].shape',expected:"(2, 4, 6)",hidden:!1},{id:"2",description:"Output shape",input:'einsum_batch_ops(np.ones((2, 4, 8)), np.ones((2, 6, 8)), np.ones((2, 6, 16)))["output"].shape',expected:"(2, 4, 16)",hidden:!1},{id:"3",description:"Attention weights and output match matmul",input:'(lambda Q, K, V: (lambda r: bool(np.allclose(r["scores"], Q @ K.transpose(0, 2, 1)) and np.allclose(r["attention_weights"].sum(-1), 1.0) and np.allclose(r["output"], r["attention_weights"] @ V)))(einsum_batch_ops(Q, K, V)))(np.random.randn(2, 3, 4), np.random.randn(2, 5, 4), np.random.randn(2, 5, 6))',expected:"True",hidden:!0}],hints:["'bqd,bkd->bqk' computes Q @ K.T for each batch","'bqk,bkv->bqv' applies attention weights to values","b is the batch dimension, preserved in output"],solution:`import numpy as np

def einsum_batch_ops(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> dict:
    # Q: (batch, seq_q, dim)
    # K: (batch, seq_k, dim)
    # V: (batch, seq_k, dim_v)

    # Attention scores: Q @ K.T for each batch
    # Result: (batch, seq_q, seq_k)
    scores = np.einsum('bqd,bkd->bqk', Q, K)

    # Scale scores
    d_k = Q.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # Softmax (simplified, along last axis)
    exp_scores = np.exp(scaled_scores - scaled_scores.max(axis=-1, keepdims=True))
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    # Result: (batch, seq_q, dim_v)
    output = np.einsum('bqk,bkv->bqv', attention_weights, V)

    return {
        'scores': scores,
        'attention_weights': attention_weights,
        'output': output
    }
`},{id:"einsum-advanced",title:"Advanced Einsum Patterns",section:"einsum",difficulty:"hard",description:`
## Advanced Einsum Patterns

Complex tensor operations for deep learning.

### Multi-Head Attention Pattern
\`\`\`python
# Split heads: (batch, seq, heads, dim)
# Attention per head: 'bhqd,bhkd->bhqk'
# Combine heads: 'bhqv->bqhv' then reshape
\`\`\`

### Bilinear Operations
\`\`\`python
# x.T @ W @ y: 'i,ijk,k->j'
# Batch bilinear: 'bi,ijk,bk->bj'
\`\`\`

### Task
Implement multi-head attention using einsum.
    `,examples:[{input:"Q, K, V with multiple heads",output:"Multi-head attention output",explanation:"Parallel attention across heads"}],starterCode:`import numpy as np

def multi_head_attention_einsum(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 num_heads: int) -> np.ndarray:
    """
    Multi-head attention using einsum.

    Args:
        Q: Queries (batch, seq_q, d_model)
        K: Keys (batch, seq_k, d_model)
        V: Values (batch, seq_k, d_model)
        num_heads: Number of attention heads

    Returns:
        output: (batch, seq_q, d_model)
    """
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape preserved",input:"multi_head_attention_einsum(np.ones((2, 4, 64)), np.ones((2, 4, 64)), np.ones((2, 4, 64)), 8).shape",expected:"(2, 4, 64)",hidden:!1},{id:"2",description:"Output is valid array",input:"not np.isnan(multi_head_attention_einsum(np.ones((2, 4, 64)), np.ones((2, 4, 64)), np.ones((2, 4, 64)), 8)).any()",expected:"True",hidden:!0}],hints:["Reshape Q, K, V to (batch, seq, num_heads, d_k)","Use 'bqhd,bkhd->bhqk' for attention scores per head","Use 'bhqk,bkhd->bqhd' for weighted values","Reshape back to (batch, seq, d_model)"],solution:`import numpy as np

def multi_head_attention_einsum(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 num_heads: int) -> np.ndarray:
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_k = d_model // num_heads

    # Reshape to (batch, seq, num_heads, d_k)
    Q = Q.reshape(batch, seq_q, num_heads, d_k)
    K = K.reshape(batch, seq_k, num_heads, d_k)
    V = V.reshape(batch, seq_k, num_heads, d_k)

    # Compute attention scores for all heads
    # (batch, seq_q, heads, d_k) x (batch, seq_k, heads, d_k) -> (batch, heads, seq_q, seq_k)
    scores = np.einsum('bqhd,bkhd->bhqk', Q, K) / np.sqrt(d_k)

    # Softmax
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    # (batch, heads, seq_q, seq_k) x (batch, seq_k, heads, d_k) -> (batch, seq_q, heads, d_k)
    output = np.einsum('bhqk,bkhd->bqhd', attention, V)

    # Reshape back to (batch, seq_q, d_model)
    output = output.reshape(batch, seq_q, d_model)

    return output
`},{id:"einsum-vs-matmul",title:"Einsum vs Traditional Operations",section:"einsum",difficulty:"easy",description:"\n## Einsum vs Traditional Operations\n\nCompare einsum with equivalent NumPy operations.\n\n### Equivalences\n| Einsum | NumPy |\n|--------|-------|\n| `'ij->'` | `np.sum(A)` |\n| `'ij->i'` | `np.sum(A, axis=1)` |\n| `'ik,kj->ij'` | `A @ B` |\n| `'ij,ij->'` | `np.sum(A * B)` |\n| `'ij->ji'` | `A.T` |\n\n### Task\nVerify einsum produces same results as traditional operations.\n\n### Expected Return Format\nReturn a dictionary with:\n- `'results'`: Dict mapping operation names to {'einsum': ..., 'numpy': ...}\n- `'all_match'`: Boolean indicating if all einsum results match numpy equivalents\n    ",examples:[{input:"A=[[1,2,3],[4,5,6]], B=[[1,2],[3,4],[5,6]]",output:"{'results': {'sum': {...}, 'matmul': {...}, ...}, 'all_match': True}",explanation:"All einsum operations produce identical results to NumPy equivalents"}],starterCode:`import numpy as np

def einsum_equivalence(A: np.ndarray, B: np.ndarray) -> dict:
    """
    Compare einsum with traditional NumPy operations.

    Args:
        A: First matrix (m, k)
        B: Second matrix (k, n)

    Returns:
        Dictionary with 'einsum' and 'numpy' results for comparison
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"All results match",input:'einsum_equivalence(np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]]), np.array([[1.0,2.0],[3.0,4.0],[5.0,6.0]]))["all_match"]',expected:"True",hidden:!1},{id:"2",description:"Matrix multiplication matches",input:'bool(np.allclose(einsum_equivalence(np.array([[1.0,2.0],[3.0,4.0]]), np.array([[5.0,6.0],[7.0,8.0]]))["results"]["matmul"]["einsum"], np.array([[19.0,22.0],[43.0,50.0]])))',expected:"True",hidden:!1},{id:"3",description:"Equivalences hold for random inputs",input:'bool(einsum_equivalence(np.random.randn(3, 4), np.random.randn(4, 3))["all_match"])',expected:"True",hidden:!0}],hints:["Use np.allclose() to compare floating point arrays","Remember 'ik,kj->ij' is matrix multiplication"],solution:`import numpy as np

def einsum_equivalence(A: np.ndarray, B: np.ndarray) -> dict:
    results = {
        'sum': {
            'einsum': np.einsum('ij->', A),
            'numpy': np.sum(A)
        },
        'row_sum': {
            'einsum': np.einsum('ij->i', A),
            'numpy': np.sum(A, axis=1)
        },
        'matmul': {
            'einsum': np.einsum('ik,kj->ij', A, B),
            'numpy': A @ B
        },
        'transpose': {
            'einsum': np.einsum('ij->ji', A),
            'numpy': A.T
        },
        'hadamard_sum': {
            'einsum': np.einsum('ij,ij->', A, A),
            'numpy': np.sum(A * A)
        }
    }

    # Verify all match
    all_match = all(
        np.allclose(v['einsum'], v['numpy'])
        for v in results.values()
    )

    return {'results': results, 'all_match': all_match}
`}],nb=[{id:"tensor-creation",title:"Tensor Creation (NumPy Style)",section:"pytorch-basics",difficulty:"easy",description:"\n## Tensor Creation\n\nLearn tensor creation patterns used in PyTorch, implemented with NumPy.\n\n### PyTorch Equivalents\n| PyTorch | NumPy |\n|---------|-------|\n| `torch.tensor([1,2,3])` | `np.array([1,2,3])` |\n| `torch.zeros(3, 4)` | `np.zeros((3, 4))` |\n| `torch.randn(2, 3)` | `np.random.randn(2, 3)` |\n| `torch.arange(10)` | `np.arange(10)` |\n| `torch.linspace(0, 1, 5)` | `np.linspace(0, 1, 5)` |\n\n### Data Types\n| PyTorch | NumPy |\n|---------|-------|\n| `torch.float32` | `np.float32` |\n| `torch.int64` | `np.int64` |\n\n### Task\nCreate tensors matching PyTorch patterns.\n\n### Expected Return Format\nReturn a dictionary with these keys:\n- `'from_list'`: Array from [1,2,3,4] with float32 dtype\n- `'zeros'`: Zero array of shape (3, 4)\n- `'ones'`: Ones array of shape (2, 3)\n- `'randn'`: Random normal array of shape (2, 3)\n- `'arange'`: Array from 0 to 9\n- `'linspace'`: 5 evenly spaced values from 0 to 1\n- `'eye'`: 4x4 identity matrix\n- `'full'`: (2, 3) array filled with 7.0\n    ",examples:[{input:"create_tensors()",output:"{'zeros': shape (3,4), 'ones': shape (2,3), 'randn': shape (2,3), 'arange': [0..9], ...}",explanation:"Dictionary containing tensors created with various methods"}],starterCode:`import numpy as np

def create_tensors() -> dict:
    """
    Create tensors using various methods.

    Returns:
        Dictionary with named tensors
    """
    np.random.seed(42)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Zeros shape",input:'create_tensors()["zeros"].shape',expected:"(3, 4)",hidden:!1},{id:"2",description:"Random tensor shape",input:'create_tensors()["randn"].shape',expected:"(2, 3)",hidden:!1},{id:"3",description:"Values of eye, arange, linspace, full",input:'(lambda r: (r["eye"].tolist(), r["arange"].tolist(), r["linspace"].tolist(), r["full"].tolist()))(create_tensors())',expected:"([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 0.25, 0.5, 0.75, 1.0], [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]])",hidden:!0}],hints:["np.zeros((3, 4)) for 3x4 zeros","np.random.randn(2, 3) for random normal","Specify dtype with astype() or dtype parameter"],solution:`import numpy as np

def create_tensors() -> dict:
    np.random.seed(42)

    return {
        'from_list': np.array([1, 2, 3, 4], dtype=np.float32),
        'zeros': np.zeros((3, 4)),
        'ones': np.ones((2, 3)),
        'randn': np.random.randn(2, 3),
        'arange': np.arange(10),
        'linspace': np.linspace(0, 1, 5),
        'eye': np.eye(4),
        'full': np.full((2, 3), 7.0),
    }
`},{id:"tensor-operations",title:"Tensor Operations",section:"pytorch-basics",difficulty:"easy",description:"\n## Basic Tensor Operations\n\nCommon tensor operations used in neural networks.\n\n### Arithmetic Operations\n- Element-wise: +, -, *, /\n- Matrix multiplication: @ or np.matmul\n- Power: ** or np.power\n\n### Reduction Operations\n- sum, mean, std, var\n- min, max, argmin, argmax\n\n### Shape Operations\n- reshape, view (reshape in NumPy)\n- squeeze, unsqueeze (np.squeeze, np.expand_dims)\n- permute, transpose\n\n### Task\nImplement common tensor operations.\n\n### Expected Return Format\nReturn a dictionary with these keys:\n- **Arithmetic**: `'add'`, `'sub'`, `'mul'`, `'div'`, `'pow'`\n- **Reductions**: `'sum_all'`, `'sum_axis0'`, `'sum_axis1'`, `'mean'`, `'std'`, `'max'`, `'argmax'`\n- **Shape ops**: `'reshape'` (to 3,2), `'flatten'`, `'unsqueeze'` (add dim 0), `'squeeze'`, `'transpose'`\n    ",examples:[{input:"x=[[1,2,3],[4,5,6]], y=[[1,1,1],[2,2,2]]",output:"{'add': [[2,3,4],[6,7,8]], 'mean': 3.5, 'reshape': [[1,2],[3,4],[5,6]], ...}",explanation:"Arithmetic, reduction, and shape operations on tensors"}],starterCode:`import numpy as np

def tensor_ops(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Perform common tensor operations.

    Args:
        x: First tensor (2, 3)
        y: Second tensor (2, 3)

    Returns:
        Dictionary with operation results
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Addition",input:'bool(np.array_equal(tensor_ops(np.array([[1,2,3],[4,5,6]]), np.array([[1,1,1],[2,2,2]]))["add"], np.array([[2,3,4],[6,7,8]])))',expected:"True",hidden:!1},{id:"2",description:"Mean computation",input:'tensor_ops(np.array([[1,2,3],[4,5,6]]), np.array([[1,1,1],[2,2,2]]))["mean"]',expected:"3.5",hidden:!1},{id:"3",description:"Reductions, argmax and reshapes",input:'(lambda r: (r["sum_axis0"].tolist(), r["sum_axis1"].tolist(), int(r["argmax"]), r["transpose"].shape, r["unsqueeze"].shape))(tensor_ops(np.array([[1, 5, 3], [4, 2, 6]]), np.array([[1, 1, 1], [2, 2, 2]])))',expected:"([5, 7, 9], [9, 12], 5, (3, 2), (1, 2, 3))",hidden:!0}],hints:["Element-wise ops work directly: x + y, x * y","Use axis parameter for reductions along specific dims","np.expand_dims(x, axis=0) adds dimension"],solution:`import numpy as np

def tensor_ops(x: np.ndarray, y: np.ndarray) -> dict:
    return {
        # Arithmetic
        'add': x + y,
        'sub': x - y,
        'mul': x * y,
        'div': x / (y + 1e-8),
        'pow': x ** 2,

        # Reductions
        'sum_all': x.sum(),
        'sum_axis0': x.sum(axis=0),
        'sum_axis1': x.sum(axis=1),
        'mean': x.mean(),
        'std': x.std(),
        'max': x.max(),
        'argmax': x.argmax(),

        # Shape ops
        'reshape': x.reshape(3, 2),
        'flatten': x.flatten(),
        'unsqueeze': np.expand_dims(x, axis=0),  # (1, 2, 3)
        'squeeze': np.squeeze(np.expand_dims(x, 0)),
        'transpose': x.T,
    }
`},{id:"autograd-concepts",title:"Autograd Concepts (Manual)",section:"pytorch-basics",difficulty:"medium",description:`
## Autograd Concepts

Understand automatic differentiation by implementing it manually.

### PyTorch Autograd
\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7
\`\`\`

### Manual Gradient Computation
For y = x² + 3x + 1:
- dy/dx = 2x + 3

### Task
Implement two functions:
1. \`compute_gradients(x)\`: forward pass and derivative of y = x² + 3x + 1. Return \`{'y': ..., 'grad': ...}\`
2. \`linear_layer_gradients(X, W, b, grad_output)\`: backward pass of Y = X @ W + b. Return \`{'grad_W': X.T @ grad_output, 'grad_b': grad_output summed over the batch, 'grad_X': grad_output @ W.T}\`
    `,examples:[{input:"compute_gradients(x=2.0)",output:"{'y': 11.0, 'grad': 7.0}",explanation:"Forward and backward pass"}],starterCode:`import numpy as np

def compute_gradients(x: float) -> dict:
    """
    Compute forward and backward pass for y = x^2 + 3x + 1.

    Args:
        x: Input value

    Returns:
        Dictionary with 'y' (forward result) and 'grad' (dy/dx)
    """
    # Your code here
    pass


def linear_layer_gradients(X: np.ndarray, W: np.ndarray, b: np.ndarray,
                           grad_output: np.ndarray) -> dict:
    """
    Compute gradients for a linear layer: Y = X @ W + b

    Args:
        X: Input (batch, in_features)
        W: Weights (in_features, out_features)
        b: Bias (out_features,)
        grad_output: Gradient from next layer (batch, out_features)

    Returns:
        Dictionary with gradients for X, W, and b
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Simple gradient",input:'compute_gradients(2.0)["grad"]',expected:"7.0",hidden:!1},{id:"2",description:"Forward pass result",input:'compute_gradients(2.0)["y"]',expected:"11.0",hidden:!1},{id:"3",description:"Linear layer gradients match finite differences",input:'(lambda X, W, b, G: (lambda r: bool(np.allclose(r["grad_W"], X.T @ G) and np.allclose(r["grad_b"], G.sum(0)) and np.allclose(r["grad_X"], G @ W.T)))(linear_layer_gradients(X, W, b, G)))(np.random.randn(4, 3), np.random.randn(3, 2), np.random.randn(2), np.random.randn(4, 2))',expected:"True",hidden:!0}],hints:["For y = x² + 3x + 1, dy/dx = 2x + 3","For Y = X @ W + b: dL/dW = X.T @ grad_output","dL/dX = grad_output @ W.T","dL/db = sum(grad_output, axis=0)"],solution:`import numpy as np

def compute_gradients(x: float) -> dict:
    # Forward pass
    y = x ** 2 + 3 * x + 1

    # Backward pass (analytical gradient)
    grad = 2 * x + 3

    return {'y': y, 'grad': grad}


def linear_layer_gradients(X: np.ndarray, W: np.ndarray, b: np.ndarray,
                           grad_output: np.ndarray) -> dict:
    # Forward: Y = X @ W + b

    # Backward:
    # dL/dW = X.T @ grad_output
    grad_W = X.T @ grad_output

    # dL/db = sum of grad_output over batch
    grad_b = grad_output.sum(axis=0)

    # dL/dX = grad_output @ W.T
    grad_X = grad_output @ W.T

    return {
        'grad_W': grad_W,
        'grad_b': grad_b,
        'grad_X': grad_X
    }
`},{id:"nn-modules",title:"Neural Network Modules",section:"pytorch-basics",difficulty:"medium",description:`
## Neural Network Modules

Implement PyTorch-style nn.Module patterns in NumPy.

### Module Pattern
\`\`\`python
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]
\`\`\`

### Task
Implement Linear and ReLU modules with forward method.
    `,examples:[{input:"Linear(10, 5).forward(x)",output:"Output of shape (batch, 5)",explanation:"Linear transformation"}],starterCode:`import numpy as np

class Linear:
    """Linear layer: Y = X @ W + b"""

    def __init__(self, in_features: int, out_features: int):
        # Initialize weights with small random values
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass

    def parameters(self):
        # Return list of parameters
        pass


class ReLU:
    """ReLU activation: max(0, x)"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass


class Sequential:
    """Container for sequential layers"""

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Your code here
        pass


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    """Create a 2-layer MLP: Linear -> ReLU -> Linear"""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Linear output shape",input:"Linear(10, 5).forward(np.random.randn(4, 10)).shape",expected:"(4, 5)",hidden:!1},{id:"2",description:"Sequential forward",input:"create_mlp(10, 20, 5).forward(np.random.randn(4, 10)).shape",expected:"(4, 5)",hidden:!1},{id:"3",description:"Linear computes x @ W + b with its own parameters",input:"(lambda l, x: bool(np.allclose(l.forward(x), x @ l.parameters()[0] + l.parameters()[1]) or np.allclose(l.forward(x), x @ l.parameters()[0].T + l.parameters()[1])))(Linear(3, 2), np.random.randn(5, 3))",expected:"True",hidden:!0}],hints:["Initialize W with np.random.randn() * 0.01","ReLU: np.maximum(0, x)","Sequential applies layers in order"],solution:`import numpy as np

class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x


def create_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    return Sequential(
        Linear(input_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, output_dim)
    )
`},{id:"loss-functions",title:"Loss Functions",section:"pytorch-basics",difficulty:"medium",description:`
## Common Loss Functions

Implement loss functions used in deep learning.

### Cross-Entropy Loss
\`\`\`
L = -sum(y * log(softmax(logits)))
\`\`\`

### MSE Loss
\`\`\`
L = mean((y_pred - y_true)²)
\`\`\`

### Binary Cross-Entropy
\`\`\`
L = -mean(y * log(p) + (1-y) * log(1-p))
\`\`\`

### Task
Implement these loss functions with numerical stability.
    `,examples:[{input:"cross_entropy(logits, targets)",output:"Scalar loss value",explanation:"Cross-entropy for classification"}],starterCode:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    # Your code here
    pass


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Cross-entropy loss for classification.

    Args:
        logits: Raw scores (batch, num_classes)
        targets: Integer class labels (batch,)

    Returns:
        Scalar loss
    """
    # Your code here
    pass


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error loss."""
    # Your code here
    pass


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary cross-entropy with numerical stability."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Cross-entropy computation",input:"round(cross_entropy_loss(np.array([[2.0, 1.0, 0.1]]), np.array([0])), 4)",expected:"0.417",hidden:!1},{id:"2",description:"MSE computation",input:"mse_loss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))",expected:"0.333333",hidden:!1},{id:"3",description:"Cross-entropy on a batch and BCE",input:"(lambda: (round(cross_entropy_loss(np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]), np.array([2, 0])), 4), round(binary_cross_entropy(np.array([0.9, 0.2]), np.array([1.0, 0.0])), 4)))()",expected:"(0.7531, 0.1643)",hidden:!0}],hints:["Softmax: exp(x - max(x)) / sum(exp(x - max(x)))","Use np.clip for numerical stability in log","Index logits with targets for cross-entropy"],solution:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    batch_size = logits.shape[0]
    probs = softmax(logits)

    # Clip for numerical stability
    probs = np.clip(probs, 1e-15, 1 - 1e-15)

    # Select probability of correct class for each sample
    correct_probs = probs[np.arange(batch_size), targets]

    # Negative log likelihood
    loss = -np.mean(np.log(correct_probs))
    return loss


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
`}],rb=[{id:"normalize-features",title:"Normalize Features",section:"data-preprocessing",difficulty:"easy",description:`
## Normalize Features (Min-Max Scaling)

Implement min-max normalization to scale features to the range [0, 1].

### Formula
\`\`\`
X_normalized = (X - X_min) / (X_max - X_min)
\`\`\`

### Function Signature
\`\`\`python
def normalize(arr: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- Input is a 1D array
- Array length: 2 <= n <= 1000
- All elements are finite numbers
    `,examples:[{input:"np.array([1, 2, 3, 4, 5])",output:"[0.0, 0.25, 0.5, 0.75, 1.0]",explanation:"min=1, max=5, so (x-1)/(5-1) gives [0, 0.25, 0.5, 0.75, 1]"}],starterCode:`import numpy as np

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to scale values to [0, 1].

    Args:
        arr: Input array of numbers

    Returns:
        Normalized array with values in [0, 1]
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic normalization",input:"[1, 2, 3, 4, 5]",expected:"[0.0, 0.25, 0.5, 0.75, 1.0]",hidden:!1},{id:"2",description:"With negative values",input:"[-10, 0, 10]",expected:"[0.0, 0.5, 1.0]",hidden:!1},{id:"3",description:"Larger range",input:"[0, 50, 100]",expected:"[0.0, 0.5, 1.0]",hidden:!0}],hints:["Use np.min() and np.max() to find the range.","Apply the formula: (x - min) / (max - min)"],solution:`import numpy as np

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to scale values to [0, 1].
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)
`},{id:"handle-missing-data",title:"Handle Missing Data",section:"data-preprocessing",difficulty:"medium",description:`
## Handle Missing Data

Replace NaN (missing) values in an array with the mean of non-NaN values.

### Function Signature
\`\`\`python
def fill_missing_with_mean(arr: np.ndarray) -> np.ndarray:
\`\`\`

### Constraints
- Input is a 1D array
- Array contains at least one non-NaN value
- NaN values are represented as np.nan
    `,examples:[{input:"np.array([1.0, np.nan, 3.0, np.nan, 5.0])",output:"[1.0, 3.0, 3.0, 3.0, 5.0]",explanation:"Mean of [1, 3, 5] is 3, which replaces NaN values"}],starterCode:`import numpy as np

def fill_missing_with_mean(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaN values with the mean of non-NaN values.

    Args:
        arr: Input array that may contain NaN values

    Returns:
        Array with NaN values replaced by the mean
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic case",input:'[1.0, float("nan"), 3.0, float("nan"), 5.0]',expected:"[1.0, 3.0, 3.0, 3.0, 5.0]",hidden:!1},{id:"2",description:"Single NaN",input:'[2.0, 4.0, float("nan"), 6.0]',expected:"[2.0, 4.0, 4.0, 6.0]",hidden:!1},{id:"3",description:"NaN at start",input:'[float("nan"), 10.0, 20.0]',expected:"[15.0, 10.0, 20.0]",hidden:!0}],hints:["Use np.isnan() to find NaN values.","Use np.nanmean() to compute mean ignoring NaN values.","Use boolean indexing to replace values."],solution:`import numpy as np

def fill_missing_with_mean(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaN values with the mean of non-NaN values.
    """
    arr = arr.copy()  # Don't modify original
    mean_val = np.nanmean(arr)
    arr[np.isnan(arr)] = mean_val
    return arr
`},{id:"one-hot-encode",title:"One-Hot Encoding",section:"data-preprocessing",difficulty:"medium",description:`
## One-Hot Encoding

Convert an array of categorical integers to one-hot encoded format.

### Function Signature
\`\`\`python
def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
\`\`\`

### Example
For labels [0, 1, 2] with 3 classes:
\`\`\`
[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1]]
\`\`\`

### Constraints
- Labels are integers in range [0, num_classes - 1]
- 1 <= num_classes <= 100
    `,examples:[{input:"labels = [0, 1, 2, 1], num_classes = 3",output:"[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]",explanation:"Each label becomes a row with 1 at the label index"}],starterCode:`import numpy as np

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded format.

    Args:
        labels: 1D array of integer labels
        num_classes: Total number of classes

    Returns:
        2D array of shape (len(labels), num_classes)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic encoding",input:"([0, 1, 2], 3)",expected:"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",hidden:!1},{id:"2",description:"Repeated labels",input:"([0, 0, 1, 1], 2)",expected:"[[1, 0], [1, 0], [0, 1], [0, 1]]",hidden:!1},{id:"3",description:"More classes than used",input:"([0, 2], 4)",expected:"[[1, 0, 0, 0], [0, 0, 1, 0]]",hidden:!0}],hints:["Create a zeros matrix of shape (num_samples, num_classes).","Use array indexing to set the appropriate positions to 1.","np.eye() can be useful for creating one-hot encodings."],solution:`import numpy as np

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded format.
    """
    # Method 1: Using np.eye
    return np.eye(num_classes)[labels].astype(int)

    # Method 2: Manual approach
    # one_hot = np.zeros((len(labels), num_classes), dtype=int)
    # one_hot[np.arange(len(labels)), labels] = 1
    # return one_hot
`}],ib=[{id:"logistic-regression",title:"Sigmoid Function",section:"supervised-learning",difficulty:"easy",description:`
## Sigmoid Function

Implement the sigmoid activation function used in logistic regression.

### Formula
\`\`\`
sigmoid(x) = 1 / (1 + exp(-x))
\`\`\`

### Properties
- Output is always between 0 and 1
- sigmoid(0) = 0.5
- Monotonically increasing

### Function Signature
\`\`\`python
def sigmoid(x: np.ndarray) -> np.ndarray:
\`\`\`
    `,examples:[{input:"np.array([0, 1, -1])",output:"[0.5, 0.731059, 0.268941]",explanation:"sigmoid(0)=0.5, sigmoid(1)≈0.73, sigmoid(-1)≈0.27"}],starterCode:`import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape with sigmoid applied element-wise
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Zero input",input:"[0]",expected:"[0.5]",hidden:!1},{id:"2",description:"Positive and negative",input:"bool(np.allclose(sigmoid(np.array([-1, 0, 1])), [0.268941, 0.5, 0.731059], atol=1e-5))",expected:"True",hidden:!1},{id:"3",description:"Large values",input:"bool(np.allclose(sigmoid(np.array([-10, 10])), [4.5e-05, 0.999955], atol=1e-5))",expected:"True",hidden:!0}],hints:["Use np.exp() for the exponential function.","The formula is 1 / (1 + exp(-x))"],solution:`import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))
`},{id:"binary-cross-entropy",title:"Binary Cross-Entropy Loss",section:"supervised-learning",difficulty:"easy",description:`
## Binary Cross-Entropy Loss

Implement the binary cross-entropy (log loss) function.

### Formula
\`\`\`
BCE = -1/m * sum(y * log(p) + (1-y) * log(1-p))
\`\`\`

Where:
- y: True labels (0 or 1)
- p: Predicted probabilities
- m: Number of samples

### Numerical Stability
Clip predictions to avoid log(0):
\`\`\`python
p = np.clip(p, 1e-15, 1 - 1e-15)
\`\`\`
    `,examples:[{input:"y = [1, 0, 1], p = [0.9, 0.1, 0.8]",output:"0.1446",explanation:"Low loss for confident correct predictions"}],starterCode:`import numpy as np

def binary_cross_entropy(y_true, y_pred):
    """
    Compute binary cross-entropy loss.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities

    Returns:
        loss: Scalar BCE loss
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect predictions",input:"([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0])",expected:"0.0",hidden:!1},{id:"2",description:"Typical case",input:"([1, 0, 1], [0.9, 0.1, 0.8])",expected:"0.1446",hidden:!1},{id:"3",description:"Mixed confident/unsure predictions",input:"([0, 1, 1, 0], [0.2, 0.7, 0.9, 0.4])",expected:"0.299",hidden:!0}],hints:["Clip predictions for numerical stability","Apply the formula element-wise","Take the mean over all samples"],solution:`import numpy as np

def binary_cross_entropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip for numerical stability
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # BCE formula
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return round(loss, 4)
`},{id:"hinge-loss",title:"Hinge Loss",section:"supervised-learning",difficulty:"easy",description:`
## Hinge Loss

Implement the hinge loss function used in Support Vector Machines (SVMs).

### Formula
\`\`\`
L = (1/n) * sum(max(0, 1 - y * s))
\`\`\`

Where:
- **y**: True labels in {-1, +1}
- **s**: Raw model scores (not probabilities)
- **n**: Number of samples

### Properties
- Loss is **0** when the prediction has the correct sign AND confidence margin >= 1
- Loss increases linearly when the prediction is wrong or within the margin
- Unlike cross-entropy, hinge loss encourages a "margin" of separation between classes

### Function Signature
\`\`\`python
def hinge_loss(y_true: np.ndarray, scores: np.ndarray) -> float:
\`\`\`

Returns the mean hinge loss rounded to 4 decimal places.
    `,examples:[{input:"y_true = [1, -1, 1], scores = [0.5, -0.8, 1.5]",output:"0.2333",explanation:"Margins: [0.5, 0.2, -0.5] → losses: [0.5, 0.2, 0.0] → mean = 0.2333"},{input:"y_true = [1, -1], scores = [2.0, -3.0]",output:"0.0",explanation:"Both predictions correct with margin >= 1, so zero loss"}],starterCode:`import numpy as np

def hinge_loss(y_true, scores):
    """
    Compute the mean hinge loss.

    Args:
        y_true: True labels in {-1, +1}, shape (n,)
        scores: Raw model scores, shape (n,)

    Returns:
        Mean hinge loss (float, rounded to 4 decimals)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect predictions with large margin",input:"([1, -1], [2.0, -3.0])",expected:"0.0",hidden:!1},{id:"2",description:"Partial margin violations",input:"([1, -1, 1], [0.5, -0.8, 1.5])",expected:"0.2333",hidden:!1},{id:"3",description:"All misclassified",input:"([1, 1], [-1.0, -2.0])",expected:"2.5",hidden:!1},{id:"4",description:"On the decision boundary (margin = 1)",input:"([1, -1], [1.0, -1.0])",expected:"0.0",hidden:!0}],hints:["Compute element-wise margins: y_true * scores","Apply max(0, 1 - margin) using np.maximum","Return the mean of the losses, rounded to 4 decimals"],solution:`import numpy as np

def hinge_loss(y_true, scores):
    """
    Compute the mean hinge loss.
    """
    y_true = np.array(y_true, dtype=float)
    scores = np.array(scores, dtype=float)

    # Compute hinge loss for each sample
    losses = np.maximum(0, 1 - y_true * scores)

    return round(float(np.mean(losses)), 4)
`},{id:"decision-tree-split",title:"Gini Impurity",section:"supervised-learning",difficulty:"medium",description:`
## Gini Impurity

Calculate the Gini impurity for a set of labels. This metric is used in decision trees to determine the best split.

### Formula
\`\`\`
Gini = 1 - sum(p_i^2)
\`\`\`

Where p_i is the proportion of class i in the set.

### Properties
- Gini = 0 means pure (all same class)
- Gini = 0.5 means maximum impurity (for binary classification with equal split)

### Function Signature
\`\`\`python
def gini_impurity(labels: np.ndarray) -> float:
\`\`\`
    `,examples:[{input:"np.array([0, 0, 0, 0])",output:"0.0",explanation:"All same class, pure node"},{input:"np.array([0, 0, 1, 1])",output:"0.5",explanation:"Equal split, maximum impurity for binary"}],starterCode:`import numpy as np

def gini_impurity(labels: np.ndarray) -> float:
    """
    Calculate Gini impurity for a set of labels.

    Args:
        labels: Array of class labels (integers)

    Returns:
        Gini impurity value between 0 and 1
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Pure node",input:"[0, 0, 0, 0]",expected:"0.0",hidden:!1},{id:"2",description:"Maximum impurity",input:"[0, 0, 1, 1]",expected:"0.5",hidden:!1},{id:"3",description:"Unequal split",input:"[0, 0, 0, 1]",expected:"0.375",hidden:!0}],hints:["Count the occurrences of each class.","Calculate the proportion (probability) of each class.","Gini = 1 - sum of squared probabilities."],solution:`import numpy as np

def gini_impurity(labels: np.ndarray) -> float:
    """
    Calculate Gini impurity for a set of labels.
    """
    if len(labels) == 0:
        return 0.0

    # Count occurrences of each class
    _, counts = np.unique(labels, return_counts=True)

    # Calculate probabilities
    probabilities = counts / len(labels)

    # Gini = 1 - sum(p^2)
    return 1 - np.sum(probabilities ** 2)
`},{id:"linear-regression-gd",title:"Linear Regression with Gradient Descent",section:"supervised-learning",difficulty:"medium",description:`
## Linear Regression with Gradient Descent

Implement simple linear regression using gradient descent optimization.

### Model
\`\`\`
y = w * x + b
\`\`\`

### Gradient Descent Updates
\`\`\`
w = w - learning_rate * dw
b = b - learning_rate * db
\`\`\`

Where:
- dw = (2/n) * sum((y_pred - y) * x)
- db = (2/n) * sum(y_pred - y)

### Function Signature
\`\`\`python
def linear_regression(X: np.ndarray, y: np.ndarray,
                      learning_rate: float, iterations: int) -> tuple:
\`\`\`

Returns (w, b) - the learned weight and bias.
    `,examples:[{input:"X = [1, 2, 3, 4], y = [2, 4, 6, 8], lr = 0.1, iters = 1000",output:"w ≈ 2.0, b ≈ 0.0",explanation:"The true relationship is y = 2x, so w should be close to 2"}],starterCode:`import numpy as np

def linear_regression(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      iterations: int = 1000) -> tuple:
    """
    Train a simple linear regression model using gradient descent.

    Args:
        X: Input features (1D array)
        y: Target values (1D array)
        learning_rate: Step size for gradient descent
        iterations: Number of training iterations

    Returns:
        Tuple of (weight, bias)
    """
    # Initialize parameters
    w = 0.0
    b = 0.0
    n = len(X)

    # Your gradient descent implementation here
    pass

    return (round(w, 2), round(b, 2))
`,testCases:[{id:"1",description:"Perfect linear relationship",input:"([1, 2, 3, 4], [2, 4, 6, 8], 0.1, 1000)",expected:"(2.0, 0.0)",hidden:!1},{id:"2",description:"With intercept",input:"([1, 2, 3, 4], [3, 5, 7, 9], 0.1, 1000)",expected:"(2.0, 1.0)",hidden:!1},{id:"3",description:"Different slope",input:"([0, 1, 2, 3], [1, 4, 7, 10], 0.1, 1000)",expected:"(3.0, 1.0)",hidden:!0}],hints:["First compute predictions: y_pred = w * X + b","Then compute gradients: dw = (2/n) * sum((y_pred - y) * X)","Update weights: w = w - learning_rate * dw"],solution:`import numpy as np

def linear_regression(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      iterations: int = 1000) -> tuple:
    """
    Train a simple linear regression model using gradient descent.
    """
    w = 0.0
    b = 0.0
    n = len(X)

    for _ in range(iterations):
        # Forward pass
        y_pred = w * X + b

        # Compute gradients
        dw = (2/n) * np.sum((y_pred - y) * X)
        db = (2/n) * np.sum(y_pred - y)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return (round(w, 2), round(b, 2))
`},{id:"logistic-regression-full",title:"Logistic Regression with Gradient Descent",section:"supervised-learning",difficulty:"medium",description:`
## Logistic Regression with Gradient Descent

Implement binary logistic regression from scratch.

### Model
\`\`\`
z = X @ w + b
y_pred = sigmoid(z) = 1 / (1 + exp(-z))
\`\`\`

### Loss (Binary Cross-Entropy)
\`\`\`
L = -1/m * sum(y * log(y_pred) + (1-y) * log(1-y_pred))
\`\`\`

### Gradients
\`\`\`
dw = 1/m * X.T @ (y_pred - y)
db = 1/m * sum(y_pred - y)
\`\`\`
    `,examples:[{input:"X (100, 2), y binary labels, 1000 iterations",output:"Trained weights and bias",explanation:"Learns decision boundary separating classes"}],starterCode:`import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    """
    Train logistic regression using gradient descent.

    Args:
        X: Features (m samples, n features)
        y: Binary labels (m,)
        learning_rate: Step size
        iterations: Number of iterations

    Returns:
        w: Learned weights (n,)
        b: Learned bias (scalar)
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    # Your code here
    pass

    return np.round(w, 4), round(b, 4)
`,testCases:[{id:"1",description:"Simple separable data",input:"(lambda r: bool(np.allclose(r[0], [6.0141, 6.0141], atol=1e-3) and np.isclose(r[1], -9.1984, atol=1e-3)))(logistic_regression(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1]), 0.5, 1000))",expected:"True",hidden:!1},{id:"2",description:"Learns a separable 1D problem",input:"(lambda r: bool(r[0][0] > 0 and np.all((np.array([[-2.0], [-1.0], [1.0], [2.0]]) @ r[0] + r[1] > 0) == np.array([False, False, True, True]))))(logistic_regression(np.array([[-2.0], [-1.0], [1.0], [2.0]]), np.array([0, 0, 1, 1])))",expected:"True",hidden:!0}],hints:["Forward: z = X @ w + b, then y_pred = sigmoid(z)","Gradients: dw = (1/m) * X.T @ (y_pred - y)","Update: w = w - lr * dw, b = b - lr * db"],solution:`import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(iterations):
        # Forward pass
        z = X @ w + b
        y_pred = sigmoid(z)

        # Compute gradients
        dw = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return np.round(w, 4), round(b, 4)
`},{id:"linear-svm",title:"Linear SVM with Gradient Descent",section:"supervised-learning",difficulty:"medium",description:`
## Linear SVM with Gradient Descent

Implement a linear Support Vector Machine (SVM) classifier using gradient descent on the hinge loss with L2 regularization.

### Objective
\`\`\`
L = (λ/2) * ||w||² + (1/m) * Σ max(0, 1 - yᵢ * (Xᵢ · w + b))
\`\`\`

### Gradients
Compute gradients over all samples. For samples where the margin \`yᵢ * (Xᵢ · w + b) < 1\` (hinge loss is active):
\`\`\`
dw = λ * w - (1/m) * Σ yᵢ * Xᵢ    (sum over violating samples)
db = -(1/m) * Σ yᵢ                  (sum over violating samples)
\`\`\`

For samples where the margin >= 1:
\`\`\`
dw contribution = λ * w   (only regularization)
db contribution = 0
\`\`\`

### Function Signature
\`\`\`python
def linear_svm(X, y, learning_rate=0.01, lambda_param=0.01, iterations=1000):
\`\`\`

Returns \`(w, b)\` — the learned weight vector (rounded to 4 decimals) and bias (rounded to 4 decimals).

**Note**: Labels \`y\` are in {-1, +1}.
    `,examples:[{input:"X = [[2,1],[3,2],[-2,-1],[-3,-2]], y = [1,1,-1,-1], lr=0.01, λ=0.01, iters=1000",output:"w and b that correctly classify all points",explanation:"SVM finds a maximum-margin hyperplane separating positive and negative points"}],starterCode:`import numpy as np

def linear_svm(X, y, learning_rate=0.01, lambda_param=0.01, iterations=1000):
    """
    Train a linear SVM using gradient descent on hinge loss + L2 regularization.

    Args:
        X: Features (m samples, n features)
        y: Labels in {-1, +1}, shape (m,)
        learning_rate: Step size for gradient descent
        lambda_param: L2 regularization strength
        iterations: Number of training iterations

    Returns:
        w: Learned weights (n,), rounded to 4 decimals
        b: Learned bias (scalar), rounded to 4 decimals
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    # Your gradient descent implementation here
    pass

    return np.round(w, 4), round(b, 4)
`,testCases:[{id:"1",description:"Correctly classifies separable 2D data",input:`(lambda: (
    X := np.array([[2, 1], [3, 2], [1, 3], [-2, -1], [-3, -2], [-1, -3]]),
    y := np.array([1, 1, 1, -1, -1, -1]),
    result := linear_svm(X, y, 0.01, 0.01, 1000),
    preds := np.sign(X @ result[0] + result[1]),
    bool(np.array_equal(preds, y))
)[-1])()`,expected:"True",hidden:!1},{id:"2",description:"Correctly classifies 1D data",input:`(lambda: (
    X := np.array([[3], [4], [5], [-3], [-4], [-5]]),
    y := np.array([1, 1, 1, -1, -1, -1]),
    result := linear_svm(X, y, 0.01, 0.01, 1000),
    preds := np.sign(X @ result[0] + result[1]),
    bool(np.array_equal(preds, y))
)[-1])()`,expected:"True",hidden:!1},{id:"3",description:"Weight vector has correct sign",input:`(lambda: (
    X := np.array([[1], [-1]]),
    y := np.array([1, -1]),
    result := linear_svm(X, y, 0.1, 0.01, 500),
    bool(result[0][0] > 0)
)[-1])()`,expected:"True",hidden:!0}],hints:["Compute margins for all samples: margins = y * (X @ w + b)","Find violating samples with mask = margins < 1","Gradient: dw = lambda_param * w - (1/m) * X[mask].T @ y[mask]","Update w and b using gradient descent"],solution:`import numpy as np

def linear_svm(X, y, learning_rate=0.01, lambda_param=0.01, iterations=1000):
    """
    Train a linear SVM using gradient descent on hinge loss + L2 regularization.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(iterations):
        # Compute margins
        margins = y * (X @ w + b)

        # Find samples violating the margin
        mask = margins < 1

        # Compute gradients
        dw = lambda_param * w - (1/m) * (X[mask].T @ y[mask])
        db = -(1/m) * np.sum(y[mask])

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

    return np.round(w, 4), round(b, 4)
`}],ob=[{id:"kmeans-clustering",title:"K-Means: Assign to Nearest Centroid",section:"unsupervised-learning",difficulty:"medium",description:`
## K-Means: Assign Points to Nearest Centroid

Implement the assignment step of K-means clustering: given data points and centroids, assign each point to its nearest centroid.

### Function Signature
\`\`\`python
def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
\`\`\`

### Args
- X: Data points of shape (n_samples, n_features)
- centroids: Cluster centers of shape (k, n_features)

### Returns
- Array of cluster assignments (integers 0 to k-1)

### Distance
Use Euclidean distance: sqrt(sum((x - c)^2))
    `,examples:[{input:"X = [[0, 0], [1, 1], [10, 10]], centroids = [[0, 0], [10, 10]]",output:"[0, 0, 1]",explanation:"First two points closer to centroid 0, last point closer to centroid 1"}],starterCode:`import numpy as np

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each point to its nearest centroid.

    Args:
        X: Data points of shape (n_samples, n_features)
        centroids: Cluster centers of shape (k, n_features)

    Returns:
        Array of cluster assignments (shape: n_samples)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Simple 2D case",input:"([[0, 0], [1, 1], [10, 10]], [[0, 0], [10, 10]])",expected:"[0, 0, 1]",hidden:!1},{id:"2",description:"Three clusters",input:"([[0, 0], [5, 5], [10, 10]], [[0, 0], [5, 5], [10, 10]])",expected:"[0, 1, 2]",hidden:!1},{id:"3",description:"Tiebreaker (first centroid wins)",input:"([[5, 5]], [[0, 0], [10, 10]])",expected:"[0]",hidden:!0}],hints:["Calculate distance from each point to each centroid.","Use np.linalg.norm() for Euclidean distance.","Use np.argmin() to find the closest centroid."],solution:`import numpy as np

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each point to its nearest centroid.
    """
    # Calculate distances from each point to each centroid
    # Using broadcasting: X[:, np.newaxis] has shape (n, 1, d)
    # centroids has shape (k, d)
    # Difference has shape (n, k, d)
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    # Return index of minimum distance for each point
    return np.argmin(distances, axis=1)
`},{id:"pca-implementation",title:"PCA: Center Data",section:"unsupervised-learning",difficulty:"easy",description:`
## PCA: Center the Data

The first step in PCA is to center the data by subtracting the mean of each feature.

### Function Signature
\`\`\`python
def center_data(X: np.ndarray) -> np.ndarray:
\`\`\`

### Args
- X: Data matrix of shape (n_samples, n_features)

### Returns
- Centered data where each column has mean 0

### Why Center?
Centering ensures PCA finds directions of maximum variance from the origin, not from an arbitrary point.
    `,examples:[{input:"X = [[1, 2], [3, 4], [5, 6]]",output:"[[-2, -2], [0, 0], [2, 2]]",explanation:"Column means are [3, 4], so we subtract them"}],starterCode:`import numpy as np

def center_data(X: np.ndarray) -> np.ndarray:
    """
    Center the data by subtracting the mean of each feature.

    Args:
        X: Data matrix of shape (n_samples, n_features)

    Returns:
        Centered data matrix of same shape
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic centering",input:"[[1, 2], [3, 4], [5, 6]]",expected:"[[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]]",hidden:!1},{id:"2",description:"Already centered",input:"[[-1, -1], [0, 0], [1, 1]]",expected:"[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]",hidden:!1},{id:"3",description:"Single column",input:"[[10], [20], [30]]",expected:"[[-10.0], [0.0], [10.0]]",hidden:!0}],hints:["Calculate the mean of each column using np.mean() with axis=0.","Subtract the mean from the data (broadcasting handles this)."],solution:`import numpy as np

def center_data(X: np.ndarray) -> np.ndarray:
    """
    Center the data by subtracting the mean of each feature.
    """
    mean = np.mean(X, axis=0)
    return X - mean
`}],ab=[{id:"precision-recall-f1",title:"Precision and Recall",section:"model-evaluation",difficulty:"medium",description:`
## Precision and Recall

Calculate precision and recall from true labels and predictions.

### Formulas
\`\`\`
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
\`\`\`

Where:
- TP = True Positives (predicted 1, actual 1)
- FP = False Positives (predicted 1, actual 0)
- FN = False Negatives (predicted 0, actual 1)

### Function Signature
\`\`\`python
def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
\`\`\`

Returns (precision, recall) rounded to 4 decimal places.
    `,examples:[{input:"y_true = [1, 1, 0, 1, 0], y_pred = [1, 0, 0, 1, 1]",output:"(0.6667, 0.6667)",explanation:"TP=2, FP=1, FN=1. Precision=2/3, Recall=2/3"}],starterCode:`import numpy as np

def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculate precision and recall for binary classification.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Tuple of (precision, recall) rounded to 4 decimal places
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Mixed predictions",input:"([1, 1, 0, 1, 0], [1, 0, 0, 1, 1])",expected:"(0.6667, 0.6667)",hidden:!1},{id:"2",description:"Perfect predictions",input:"([1, 1, 0, 0], [1, 1, 0, 0])",expected:"(1.0, 1.0)",hidden:!1},{id:"3",description:"High precision, low recall",input:"([1, 1, 1, 1, 0], [1, 0, 0, 0, 0])",expected:"(1.0, 0.25)",hidden:!0}],hints:["Count TP where both y_true and y_pred are 1.","Count FP where y_pred is 1 but y_true is 0.","Count FN where y_true is 1 but y_pred is 0."],solution:`import numpy as np

def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculate precision and recall for binary classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate TP, FP, FN
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return (round(precision, 4), round(recall, 4))
`},{id:"cross-validation",title:"K-Fold Split Indices",section:"model-evaluation",difficulty:"medium",description:`
## K-Fold Cross-Validation Indices

Generate train and validation indices for K-fold cross-validation.

### Concept
Split data into K equal parts. For each fold:
- Use 1 part for validation
- Use remaining K-1 parts for training

### Function Signature
\`\`\`python
def kfold_indices(n_samples: int, k: int) -> list:
\`\`\`

Returns list of (train_indices, val_indices) tuples for each fold.
    `,examples:[{input:"n_samples = 6, k = 3",output:"[([2,3,4,5], [0,1]), ([0,1,4,5], [2,3]), ([0,1,2,3], [4,5])]",explanation:"Split [0,1,2,3,4,5] into 3 folds of size 2"}],starterCode:`import numpy as np

def kfold_indices(n_samples: int, k: int) -> list:
    """
    Generate train/validation indices for K-fold cross-validation.

    Args:
        n_samples: Total number of samples
        k: Number of folds

    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic 3-fold",input:"(6, 3)",expected:"[[[2, 3, 4, 5], [0, 1]], [[0, 1, 4, 5], [2, 3]], [[0, 1, 2, 3], [4, 5]]]",hidden:!1},{id:"2",description:"2-fold",input:"(4, 2)",expected:"[[[2, 3], [0, 1]], [[0, 1], [2, 3]]]",hidden:!1},{id:"3",description:"4-fold",input:"(8, 4)",expected:"[[[2, 3, 4, 5, 6, 7], [0, 1]], [[0, 1, 4, 5, 6, 7], [2, 3]], [[0, 1, 2, 3, 6, 7], [4, 5]], [[0, 1, 2, 3, 4, 5], [6, 7]]]",hidden:!0}],hints:["Divide indices into k equal-sized chunks.","For each fold, one chunk is validation, rest are training.","Use np.array_split() to create chunks."],solution:`import numpy as np

def kfold_indices(n_samples: int, k: int) -> list:
    """
    Generate train/validation indices for K-fold cross-validation.
    """
    indices = np.arange(n_samples)
    fold_sizes = np.full(k, n_samples // k)
    fold_sizes[:n_samples % k] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        val_indices = list(range(current, current + fold_size))
        train_indices = [i for i in range(n_samples) if i not in val_indices]
        folds.append((train_indices, val_indices))
        current += fold_size

    return folds
`},{id:"confusion-matrix",title:"Accuracy Score",section:"model-evaluation",difficulty:"easy",description:`
## Accuracy Score

Calculate the accuracy of predictions - the proportion of correct predictions.

### Formula
\`\`\`
Accuracy = (Number of correct predictions) / (Total predictions)
\`\`\`

### Function Signature
\`\`\`python
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
\`\`\`

### Note
While simple, accuracy can be misleading for imbalanced datasets.
    `,examples:[{input:"y_true = [0, 1, 1, 0, 1], y_pred = [0, 1, 0, 0, 1]",output:"0.8",explanation:"4 correct out of 5 predictions"}],starterCode:`import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score between 0 and 1
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic accuracy",input:"([0, 1, 1, 0, 1], [0, 1, 0, 0, 1])",expected:"0.8",hidden:!1},{id:"2",description:"Perfect accuracy",input:"([1, 1, 0, 0], [1, 1, 0, 0])",expected:"1.0",hidden:!1},{id:"3",description:"Zero accuracy",input:"([0, 0, 1, 1], [1, 1, 0, 0])",expected:"0.0",hidden:!0}],hints:["Compare y_true and y_pred element-wise.","Count matches and divide by total.","np.mean() of boolean array gives proportion of True."],solution:`import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)
`}],sb=[{id:"perceptron",title:"ReLU Activation",section:"neural-networks",difficulty:"easy",description:`
## ReLU Activation Function

Implement the Rectified Linear Unit (ReLU) activation function, one of the most popular activation functions in deep learning.

### Formula
\`\`\`
ReLU(x) = max(0, x)
\`\`\`

### Properties
- Output is x if x > 0, else 0
- Helps with vanishing gradient problem
- Computationally efficient

### Function Signature
\`\`\`python
def relu(x: np.ndarray) -> np.ndarray:
\`\`\`
    `,examples:[{input:"np.array([-2, -1, 0, 1, 2])",output:"[0, 0, 0, 1, 2]",explanation:"Negative values become 0, positive values unchanged"}],starterCode:`import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU activation function.

    Args:
        x: Input array of any shape

    Returns:
        Array with ReLU applied element-wise
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Mixed values",input:"[-2, -1, 0, 1, 2]",expected:"[0, 0, 0, 1, 2]",hidden:!1},{id:"2",description:"All negative",input:"[-5, -3, -1]",expected:"[0, 0, 0]",hidden:!1},{id:"3",description:"All positive",input:"[1, 2, 3]",expected:"[1, 2, 3]",hidden:!0}],hints:["Use np.maximum() to compare with 0.","Alternatively, use boolean indexing or np.where()."],solution:`import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU activation function.
    """
    return np.maximum(0, x)
`},{id:"neural-network-forward",title:"Dense Layer Forward Pass",section:"neural-networks",difficulty:"medium",description:`
## Dense Layer Forward Pass

Implement the forward pass of a fully-connected (dense) neural network layer.

### Formula
\`\`\`
output = activation(X @ W + b)
\`\`\`

Where:
- X: Input of shape (batch_size, input_dim)
- W: Weights of shape (input_dim, output_dim)
- b: Bias of shape (output_dim,)

### Function Signature
\`\`\`python
def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
\`\`\`

Use ReLU as the activation function.
    `,examples:[{input:"X = [[1, 2]], W = [[1, 0], [0, 1]], b = [1, 1]",output:"[[2, 3]]",explanation:"X @ W = [[1, 2]], + b = [[2, 3]], ReLU keeps positive"}],starterCode:`import numpy as np

def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute forward pass of a dense layer with ReLU activation.

    Args:
        X: Input of shape (batch_size, input_dim)
        W: Weights of shape (input_dim, output_dim)
        b: Bias of shape (output_dim,)

    Returns:
        Output of shape (batch_size, output_dim)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic forward pass",input:"([[1, 2]], [[1, 0], [0, 1]], [1, 1])",expected:"[[2, 3]]",hidden:!1},{id:"2",description:"With negative pre-activation",input:"([[1, 1]], [[1, -2], [-1, 1]], [0, 0])",expected:"[[0, 0]]",hidden:!1},{id:"3",description:"Batch of 2",input:"([[1, 0], [0, 1]], [[2, 0], [0, 2]], [0, 0])",expected:"[[2, 0], [0, 2]]",hidden:!0}],hints:["First compute the linear transformation: Z = X @ W + b","Then apply ReLU: output = max(0, Z)"],solution:`import numpy as np

def dense_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute forward pass of a dense layer with ReLU activation.
    """
    # Linear transformation
    Z = np.dot(X, W) + b
    # ReLU activation
    return np.maximum(0, Z)
`},{id:"backpropagation",title:"Softmax Function",section:"neural-networks",difficulty:"medium",description:`
## Softmax Function

Implement the softmax function, used to convert raw scores to probabilities in multi-class classification.

### Formula
\`\`\`
softmax(x)_i = exp(x_i) / sum(exp(x_j))
\`\`\`

### Properties
- Output sums to 1
- All outputs are positive
- Larger inputs get larger probabilities

### Function Signature
\`\`\`python
def softmax(x: np.ndarray) -> np.ndarray:
\`\`\`

**Note**: For numerical stability, subtract max(x) before computing exp.
    `,examples:[{input:"np.array([1, 2, 3])",output:"[0.090031, 0.244728, 0.665241]",explanation:"Higher values get higher probabilities, sum = 1"}],starterCode:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.

    Args:
        x: Input array (1D)

    Returns:
        Array of same shape with softmax probabilities
    """
    # Your code here
    # Hint: subtract max for numerical stability
    pass
`,testCases:[{id:"1",description:"Basic softmax",input:"bool(np.allclose(softmax(np.array([1, 2, 3])), [0.090031, 0.244728, 0.665241], atol=1e-5))",expected:"True",hidden:!1},{id:"2",description:"Equal inputs",input:"bool(np.allclose(softmax(np.array([1, 1, 1])), [0.333333, 0.333333, 0.333333], atol=1e-5))",expected:"True",hidden:!1},{id:"3",description:"Large values",input:"bool(np.allclose(softmax(np.array([100, 101, 102])), [0.090031, 0.244728, 0.665241], atol=1e-5))",expected:"True",hidden:!0}],hints:["Subtract the max value for numerical stability.","Compute exp of each element.","Divide by the sum of all exp values."],solution:`import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)
`},{id:"mlp-forward-backward",title:"MLP Forward & Backward Pass",section:"neural-networks",difficulty:"hard",description:"\n## MLP Forward & Backward Pass\n\nImplement both forward and backward passes for a 2-layer MLP (Multi-Layer Perceptron).\n\n### Architecture\n```\nInput (n_features) → Hidden (n_hidden) → Output (n_classes)\n```\n\n### Forward Pass\n```\nZ1 = X @ W1 + b1\nA1 = ReLU(Z1)\nZ2 = A1 @ W2 + b2\noutput = softmax(Z2)\n```\n\nReturn: `(output, cache)` where cache = `(X, Z1, A1, Z2, W1, W2)`\n\n### Backward Pass (Backpropagation)\nUsing the chain rule, gradients flow backwards:\n\n```\nLoss → dZ2 → dW2, db2 → dA1 → dZ1 (ReLU) → dW1, db1\n```\n\n**Key Formulas:**\n- `dZ2 = output - Y_onehot` (softmax + cross-entropy derivative)\n- `dW2 = (1/m) * A1.T @ dZ2`\n- `db2 = (1/m) * sum(dZ2, axis=0)`\n- `dZ1 = (dZ2 @ W2.T) * (Z1 > 0)` (ReLU derivative)\n- `dW1 = (1/m) * X.T @ dZ1`\n- `db1 = (1/m) * sum(dZ1, axis=0)`\n\nReturn: `{'dW1': ..., 'db1': ..., 'dW2': ..., 'db2': ...}`\n    ",examples:[{input:"X shape (2, 3), hidden_size=4, output_size=2",output:"Forward: probabilities (2, 2), rows sum to 1. Backward: gradients for all weights/biases",explanation:"Forward computes predictions; backward computes gradients for training"}],starterCode:`import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mlp_forward(X, W1, b1, W2, b2):
    """
    Forward pass for a 2-layer MLP.

    Args:
        X: Input data (batch_size, n_features)
        W1: First layer weights (n_features, n_hidden)
        b1: First layer bias (n_hidden,)
        W2: Second layer weights (n_hidden, n_classes)
        b2: Second layer bias (n_classes,)

    Returns:
        output: Class probabilities (batch_size, n_classes)
        cache: Tuple (X, Z1, A1, Z2, W1, W2) for backprop
    """
    # Your code here
    pass

def mlp_backward(Y, output, cache):
    """
    Backward pass (backpropagation) for 2-layer MLP.

    Args:
        Y: One-hot encoded labels (batch_size, n_classes)
        output: Predicted probabilities from forward pass
        cache: Cached values from forward pass (X, Z1, A1, Z2, W1, W2)

    Returns:
        grads: Dictionary with dW1, db1, dW2, db2
    """
    X, Z1, A1, Z2, W1, W2 = cache
    m = X.shape[0]  # batch size

    # Your code here
    pass
`,testCases:[{id:"1",description:"Forward: output shape correct",input:"mlp_forward(np.array([[1, 2, 3], [4, 5, 6]]), np.ones((3, 4)), np.zeros(4), np.ones((4, 2)), np.zeros(2))[0].shape",expected:"(2, 2)",hidden:!1},{id:"2",description:"Forward: probabilities sum to 1",input:"round(float(np.sum(mlp_forward(np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.zeros(2), np.array([[1, 0], [0, 1]]), np.zeros(2))[0][0])), 1)",expected:"1.0",hidden:!1},{id:"3",description:"Backward: dW2 shape matches W2",input:`(lambda: (
            X := np.array([[1.0, 2.0], [3.0, 4.0]]),
            W1 := np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            b1 := np.zeros(3),
            W2 := np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            b2 := np.zeros(2),
            result := mlp_forward(X, W1, b1, W2, b2),
            output := result[0],
            cache := result[1],
            Y := np.array([[1, 0], [0, 1]]),
            grads := mlp_backward(Y, output, cache),
            grads['dW2'].shape == W2.shape
        )[-1])()`,expected:"True",hidden:!1},{id:"4",description:"Backward: gradients are non-zero",input:`(lambda: (
            X := np.array([[1.0, 2.0], [3.0, 4.0]]),
            W1 := np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            b1 := np.zeros(3),
            W2 := np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            b2 := np.zeros(2),
            result := mlp_forward(X, W1, b1, W2, b2),
            output := result[0],
            cache := result[1],
            Y := np.array([[1, 0], [0, 1]]),
            grads := mlp_backward(Y, output, cache),
            bool(np.any(grads['dW1'] != 0) and np.any(grads['dW2'] != 0))
        )[-1])()`,expected:"True",hidden:!0},{id:"5",description:"Backward: all gradient keys present",input:`(lambda: (
            X := np.array([[1.0, 2.0]]),
            W1 := np.array([[0.1, 0.2], [0.3, 0.4]]),
            b1 := np.zeros(2),
            W2 := np.array([[0.1], [0.2]]),
            b2 := np.zeros(1),
            result := mlp_forward(X, W1, b1, W2, b2),
            grads := mlp_backward(np.array([[1]]), result[0], result[1]),
            sorted(grads.keys())
        )[-1])()`,expected:'["dW1", "dW2", "db1", "db2"]',hidden:!0}],hints:["Forward: Compute Z1 = X @ W1 + b1, then A1 = ReLU(Z1)","Forward: Compute Z2 = A1 @ W2 + b2, then output = softmax(Z2)","Forward: Store cache = (X, Z1, A1, Z2, W1, W2) for backprop","Backward: Start from output layer: dZ2 = output - Y","Backward: dW2 = (1/m) * A1.T @ dZ2, db2 = (1/m) * sum(dZ2, axis=0)","Backward: Propagate through ReLU: dZ1 = (dZ2 @ W2.T) * (Z1 > 0)","Backward: dW1 = (1/m) * X.T @ dZ1, db1 = (1/m) * sum(dZ1, axis=0)"],solution:`import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mlp_forward(X, W1, b1, W2, b2):
    # First layer
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # Second layer
    Z2 = A1 @ W2 + b2
    output = softmax(Z2)

    # Cache for backprop
    cache = (X, Z1, A1, Z2, W1, W2)

    return output, cache

def mlp_backward(Y, output, cache):
    X, Z1, A1, Z2, W1, W2 = cache
    m = X.shape[0]

    # Output layer gradient (softmax + cross-entropy)
    dZ2 = output - Y
    dW2 = (1/m) * A1.T @ dZ2
    db2 = (1/m) * np.sum(dZ2, axis=0)

    # Hidden layer gradient
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)  # ReLU derivative
    dW1 = (1/m) * X.T @ dZ1
    db1 = (1/m) * np.sum(dZ1, axis=0)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
`},{id:"cross-entropy-loss",title:"Cross-Entropy Loss",section:"neural-networks",difficulty:"easy",description:`
## Cross-Entropy Loss

Implement the cross-entropy loss function for multi-class classification.

### Formula
\`\`\`
L = -1/m * sum(Y * log(Y_pred))
\`\`\`

Where:
- Y is one-hot encoded true labels
- Y_pred is predicted probabilities
- m is batch size

### Numerical Stability
Add small epsilon to avoid log(0):
\`\`\`python
np.log(Y_pred + 1e-15)
\`\`\`
    `,examples:[{input:"Y_pred = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], Y = [[1,0,0], [0,1,0]]",output:"0.2899",explanation:"-1/2 * (log(0.7) + log(0.8))"}],starterCode:`import numpy as np

def cross_entropy_loss(Y_pred, Y_true):
    """
    Compute cross-entropy loss.

    Args:
        Y_pred: Predicted probabilities (batch_size, n_classes)
        Y_true: One-hot encoded true labels (batch_size, n_classes)

    Returns:
        loss: Scalar cross-entropy loss
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect prediction",input:"abs(cross_entropy_loss(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1, 0], [0, 1]])))",expected:"0.0",hidden:!1},{id:"2",description:"Typical case",input:"cross_entropy_loss(np.array([[0.7, 0.3], [0.2, 0.8]]), np.array([[1, 0], [0, 1]]))",expected:"0.2899",hidden:!1},{id:"3",description:"Three classes, batch of 3",input:"cross_entropy_loss(np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.25, 0.25, 0.5]]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))",expected:"0.5365",hidden:!0}],hints:["Use np.log with a small epsilon for numerical stability","Multiply element-wise: Y_true * np.log(Y_pred)","Sum and negate, then divide by batch size"],solution:`import numpy as np

def cross_entropy_loss(Y_pred, Y_true):
    m = Y_pred.shape[0]
    epsilon = 1e-15
    log_probs = np.log(Y_pred + epsilon)
    loss = -np.sum(Y_true * log_probs) / m
    return round(loss, 4)
`},{id:"weight-init",title:"Xavier/He Weight Initialization",section:"neural-networks",difficulty:"medium",description:`
## Weight Initialization

Implement Xavier and He weight initialization to prevent vanishing/exploding gradients.

### Xavier Initialization (for tanh/sigmoid)
\`\`\`
W = randn(n_in, n_out) * sqrt(2 / (n_in + n_out))
\`\`\`

### He Initialization (for ReLU)
\`\`\`
W = randn(n_in, n_out) * sqrt(2 / n_in)
\`\`\`

### Why It Matters
- Keeps variance of activations stable across layers
- Prevents gradients from vanishing or exploding
- Enables training of deep networks
    `,examples:[{input:'n_in=784, n_out=256, method="he"',output:"Weights with std ≈ 0.0505",explanation:"sqrt(2/784) ≈ 0.0505"}],starterCode:`import numpy as np

def initialize_weights(n_in, n_out, method='xavier'):
    """
    Initialize weights using Xavier or He initialization.

    Args:
        n_in: Number of input units
        n_out: Number of output units
        method: 'xavier' or 'he'

    Returns:
        std: Standard deviation of initialized weights (rounded to 4 decimals)
    """
    np.random.seed(42)  # For reproducibility
    # Your code here
    pass
`,testCases:[{id:"1",description:"Xavier std correct",input:'(100, 50, "xavier")',expected:"0.1151",hidden:!1},{id:"2",description:"He std correct",input:'(100, 50, "he")',expected:"0.1409",hidden:!1},{id:"3",description:"He init for a wide layer",input:'(512, 256, "he")',expected:"0.0625",hidden:!0}],hints:["Use np.random.randn to generate random numbers","Xavier: multiply by sqrt(2 / (n_in + n_out))","He: multiply by sqrt(2 / n_in)"],solution:`import numpy as np

def initialize_weights(n_in, n_out, method='xavier'):
    np.random.seed(42)

    if method == 'xavier':
        std = np.sqrt(2.0 / (n_in + n_out))
    elif method == 'he':
        std = np.sqrt(2.0 / n_in)
    else:
        raise ValueError("Method must be 'xavier' or 'he'")

    W = np.random.randn(n_in, n_out) * std
    return round(np.std(W), 4)
`},{id:"batch-norm",title:"Batch Normalization",section:"neural-networks",difficulty:"medium",description:`
## Batch Normalization

Implement batch normalization to stabilize training.

### Forward Pass
\`\`\`
1. Compute batch mean: μ = mean(x, axis=0)
2. Compute batch variance: σ² = var(x, axis=0)
3. Normalize: x_norm = (x - μ) / sqrt(σ² + ε)
4. Scale and shift: out = γ * x_norm + β
\`\`\`

### Return Format
Return \`(out, cache)\` where:
- \`out\`: Normalized and scaled output
- \`cache\`: Tuple \`(X, X_norm, mu, var, gamma, eps)\` for backward pass

### Benefits
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization
    `,examples:[{input:"X with mean=5, var=4, gamma=1, beta=0",output:"Normalized X with mean≈0, var≈1",explanation:"BatchNorm normalizes each feature"}],starterCode:`import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    """
    Batch normalization forward pass.

    Args:
        X: Input (batch_size, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant for numerical stability

    Returns:
        out: Normalized output (batch_size, features)
        cache: Tuple (X, X_norm, mu, var, gamma, eps) for backward pass
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output mean near zero",input:"bool(np.allclose(np.mean(batch_norm_forward(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.ones(2), np.zeros(2))[0], axis=0), 0, atol=1e-5))",expected:"True",hidden:!1},{id:"2",description:"Gamma scales output",input:"list(np.round(batch_norm_forward(np.array([[0.0, 0.0], [2.0, 2.0]]), np.array([2.0, 3.0]), np.zeros(2))[0][1], 1))",expected:"[2.0, 3.0]",hidden:!0}],hints:["Compute mean along axis=0 (batch dimension)","Compute variance along axis=0","Normalize: (X - mean) / sqrt(var + eps)","Apply scale and shift: gamma * normalized + beta"],solution:`import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    # Compute batch statistics
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    # Normalize
    X_norm = (X - mu) / np.sqrt(var + eps)

    # Scale and shift
    out = gamma * X_norm + beta

    # Cache for backward pass
    cache = (X, X_norm, mu, var, gamma, eps)

    return out, cache
`},{id:"dropout",title:"Dropout",section:"neural-networks",difficulty:"easy",description:`
## Dropout Regularization

Implement dropout to prevent overfitting.

### Training Mode
\`\`\`
1. Generate mask: mask = (rand < keep_prob) / keep_prob
2. Apply mask: out = x * mask
\`\`\`

### Key Points
- Randomly "drop" neurons with probability (1 - keep_prob)
- Scale by 1/keep_prob to maintain expected value
- Disabled during inference (return input unchanged)
    `,examples:[{input:"X = [[1, 2, 3, 4]], keep_prob=0.5",output:"Some values zeroed, others scaled by 2",explanation:"Half neurons dropped, rest doubled"}],starterCode:`import numpy as np

def dropout_forward(X, keep_prob=0.5, training=True):
    """
    Apply dropout to input.

    Args:
        X: Input array
        keep_prob: Probability of keeping a neuron
        training: Whether in training mode

    Returns:
        out: Output after dropout
        mask: Dropout mask (for backward pass)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Inference returns unchanged",input:"dropout_forward(np.array([[1, 2, 3, 4]]), 0.5, False)[0].tolist()",expected:"[[1, 2, 3, 4]]",hidden:!1},{id:"2",description:"Expected value preserved",input:`(lambda: (
            np.random.seed(42),
            X := np.ones((1000, 100)),
            out := dropout_forward(X, 0.8, True)[0],
            bool(abs(np.mean(out) - 1.0) < 0.1)
        )[-1])()`,expected:"True",hidden:!0}],hints:["In training: create binary mask with np.random.rand","Scale mask by 1/keep_prob (inverted dropout)","In inference: return input unchanged"],solution:`import numpy as np

def dropout_forward(X, keep_prob=0.5, training=True):
    if not training:
        return X, None

    # Create mask and scale
    mask = (np.random.rand(*X.shape) < keep_prob) / keep_prob
    out = X * mask

    return out, mask
`}],lb=[],ub=[{id:"conv2d-forward",title:"2D Convolution",section:"cnn",difficulty:"hard",description:`
## 2D Convolution Operation

Implement the forward pass of a 2D convolution (no padding, stride=1).

### Operation
For each position (i, j) in the output:
\`\`\`
out[i, j] = sum(input[i:i+kH, j:j+kW] * kernel)
\`\`\`

### Output Size
\`\`\`
out_height = input_height - kernel_height + 1
out_width = input_width - kernel_width + 1
\`\`\`

### Function Signature
\`\`\`python
def conv2d(image, kernel):
    # image: (H, W)
    # kernel: (kH, kW)
    # output: (H-kH+1, W-kW+1)
\`\`\`
    `,examples:[{input:"image 4x4, kernel 3x3",output:"output 2x2",explanation:"4-3+1 = 2 in each dimension"}],starterCode:`import numpy as np

def conv2d(image, kernel):
    """
    Apply 2D convolution to an image.

    Args:
        image: Input image (H, W)
        kernel: Convolution kernel (kH, kW)

    Returns:
        output: Convolved image (H-kH+1, W-kW+1)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Identity kernel",input:"([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]])",expected:"[[5]]",hidden:!1},{id:"2",description:"Edge detection",input:"([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])",expected:"[[5, 5], [5, 5]]",hidden:!1},{id:"3",description:"Asymmetric kernel on a 4x4 image",input:"([[1, 2, 3, 0], [4, 1, 0, 2], [0, 5, 1, 3], [2, 0, 6, 1]], [[1, 0], [2, -1]])",expected:"[[8, 4, 1], [-1, 10, -1], [4, -1, 12]]",hidden:!0}],hints:["Use nested loops to slide the kernel over the image","At each position, compute element-wise product and sum","Output size is (H-kH+1, W-kW+1)"],solution:`import numpy as np

def conv2d(image, kernel):
    image = np.array(image)
    kernel = np.array(kernel)

    H, W = image.shape
    kH, kW = kernel.shape

    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i:i+kH, j:j+kW]
            output[i, j] = np.sum(region * kernel)

    return output.astype(int).tolist()
`},{id:"max-pool",title:"Max Pooling",section:"cnn",difficulty:"medium",description:`
## Max Pooling

Implement 2x2 max pooling with stride 2.

### Operation
Divide input into non-overlapping 2x2 regions and take maximum of each.

### Output Size
\`\`\`
out_height = input_height // 2
out_width = input_width // 2
\`\`\`

### Purpose
- Reduces spatial dimensions
- Provides translation invariance
- Reduces computation
    `,examples:[{input:"[[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]",output:"[[6, 8], [14, 16]]",explanation:"Max of each 2x2 region"}],starterCode:`import numpy as np

def max_pool2d(image, pool_size=2):
    """
    Apply 2D max pooling.

    Args:
        image: Input image (H, W)
        pool_size: Size of pooling window

    Returns:
        output: Pooled image (H//pool_size, W//pool_size)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"4x4 to 2x2",input:"[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",expected:"[[6, 8], [14, 16]]",hidden:!1},{id:"2",description:"With negative values",input:"[[-1, -2, -3, -4], [-5, -6, -7, -8], [-9, -10, -11, -12], [-13, -14, -15, -16]]",expected:"[[-1, -3], [-9, -11]]",hidden:!0}],hints:["Iterate with step size = pool_size","For each 2x2 region, use np.max()","Output dimensions are input dimensions // pool_size"],solution:`import numpy as np

def max_pool2d(image, pool_size=2):
    image = np.array(image)
    H, W = image.shape

    out_H = H // pool_size
    out_W = W // pool_size

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i*pool_size:(i+1)*pool_size,
                          j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(region)

    return output.astype(int).tolist()
`},{id:"flatten-layer",title:"Flatten Layer",section:"cnn",difficulty:"easy",description:`
## Flatten Layer

Implement the flatten operation that converts a 3D feature map to a 1D vector for the fully connected layer.

### Operation
\`\`\`
(batch, height, width, channels) → (batch, height * width * channels)
\`\`\`

### Usage
- Connects convolutional layers to fully connected layers
- Preserves batch dimension
    `,examples:[{input:"shape (2, 4, 4, 3)",output:"shape (2, 48)",explanation:"4 * 4 * 3 = 48 features per sample"}],starterCode:`import numpy as np

def flatten(X):
    """
    Flatten feature maps to vectors.

    Args:
        X: Input tensor (batch, height, width, channels)

    Returns:
        output: Flattened tensor (batch, height*width*channels)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Correct output shape",input:"flatten(np.random.randn(2, 4, 4, 3)).shape",expected:"(2, 48)",hidden:!1},{id:"2",description:"Values preserved",input:"bool(np.allclose(flatten(np.arange(24).reshape(1, 2, 3, 4)), np.arange(24).reshape(1, 24)))",expected:"True",hidden:!0}],hints:["Get batch size as X.shape[0]","Use reshape to flatten remaining dimensions","np.reshape(X, (batch_size, -1)) uses -1 to infer size"],solution:`import numpy as np

def flatten(X):
    batch_size = X.shape[0]
    return X.reshape(batch_size, -1)
`},{id:"conv-output-size",title:"Convolution Output Size",section:"cnn",difficulty:"easy",description:`
## Calculate Convolution Output Size

Implement a function to calculate the output dimensions of a convolution layer.

### Formula
\`\`\`
output_size = (input_size - kernel_size + 2 * padding) / stride + 1
\`\`\`

### Parameters
- **input_size**: Height or width of input
- **kernel_size**: Height or width of kernel
- **padding**: Zero-padding added to input
- **stride**: Step size of kernel
    `,examples:[{input:"input=32, kernel=3, padding=1, stride=1",output:"32",explanation:"(32 - 3 + 2*1) / 1 + 1 = 32"}],starterCode:`def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    """
    Calculate output size of a convolution layer.

    Args:
        input_size: Input dimension (height or width)
        kernel_size: Kernel dimension
        padding: Zero-padding
        stride: Stride

    Returns:
        output_size: Output dimension
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Same padding",input:"(32, 3, 1, 1)",expected:"32",hidden:!1},{id:"2",description:"No padding, stride 2",input:"(32, 3, 0, 2)",expected:"15",hidden:!1},{id:"3",description:"Large kernel",input:"(224, 7, 3, 2)",expected:"112",hidden:!0}],hints:["Apply the formula: (input - kernel + 2*padding) / stride + 1","Use integer division (//) for the result"],solution:`def conv_output_size(input_size, kernel_size, padding=0, stride=1):
    return (input_size - kernel_size + 2 * padding) // stride + 1
`},{id:"conv2d-advanced",title:"Advanced 2D Convolution",section:"cnn",difficulty:"hard",description:`
## Advanced 2D Convolution

Implement a full-featured 2D convolution with **padding**, **stride**, and **groups** support.

### Parameters
- **padding**: Zero-padding added to input borders
- **stride**: Step size when sliding the kernel
- **groups**: Split input/output channels into groups (used in depthwise separable convolutions)

### Group Convolution
When \`groups > 1\`:
- Input channels are split into \`groups\` chunks
- Output channels are split into \`groups\` chunks
- Each group's input is convolved only with its corresponding kernel group
- \`in_channels\` and \`out_channels\` must be divisible by \`groups\`

### Special Cases
- \`groups=1\`: Standard convolution
- \`groups=in_channels=out_channels\`: Depthwise convolution (MobileNet)

### Output Size Formula
\`\`\`
H_out = (H_in + 2*padding - kernel_height) // stride + 1
W_out = (W_in + 2*padding - kernel_width) // stride + 1
\`\`\`

### Function Signature
\`\`\`python
def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    # input: (batch, in_channels, H, W)
    # kernel: (out_channels, in_channels//groups, kH, kW)
    # output: (batch, out_channels, H_out, W_out)
\`\`\`
    `,examples:[{input:"input (1, 4, 5, 5), kernel (8, 2, 3, 3), groups=2",output:"output (1, 8, 3, 3)",explanation:"Groups=2: channels 0-1 use kernels 0-3, channels 2-3 use kernels 4-7"},{input:"input (1, 1, 5, 5), kernel (1, 1, 3, 3), padding=1, stride=2",output:"output (1, 1, 3, 3)",explanation:"(5+2*1-3)//2+1 = 3"}],starterCode:`import numpy as np

def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    """
    Advanced 2D convolution with padding, stride, and groups.

    Args:
        input: Input tensor (batch, in_channels, H, W)
        kernel: Convolution kernels (out_channels, in_channels//groups, kH, kW)
        padding: Zero-padding added to input (default: 0)
        stride: Stride of the convolution (default: 1)
        groups: Number of groups for grouped convolution (default: 1)

    Returns:
        output: Convolved tensor (batch, out_channels, H_out, W_out)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic convolution with padding and stride",input:"conv2d_advanced(np.ones((1, 1, 5, 5)), np.ones((1, 1, 3, 3)), 1, 2, 1).shape",expected:"(1, 1, 3, 3)",hidden:!1},{id:"2",description:"Grouped convolution (groups=2)",input:"conv2d_advanced(np.arange(32).reshape(1, 2, 4, 4).astype(float), np.ones((4, 1, 2, 2)), 0, 1, 2).shape",expected:"(1, 4, 3, 3)",hidden:!1},{id:"3",description:"Depthwise convolution (groups=in_channels)",input:"conv2d_advanced(np.ones((2, 3, 4, 4)), np.ones((3, 1, 2, 2)), 0, 1, 3).shape",expected:"(2, 3, 3, 3)",hidden:!0},{id:"4",description:"Verify correct output values with groups",input:"bool(np.allclose(conv2d_advanced(np.ones((1, 2, 3, 3)), np.ones((2, 1, 2, 2)), 0, 1, 2), np.array([[[[4., 4.], [4., 4.]], [[4., 4.], [4., 4.]]]])) )",expected:"True",hidden:!0}],hints:["First, pad the input using np.pad() along the H and W dimensions","Calculate output dimensions: H_out = (H_padded - kH) // stride + 1","For groups: split input channels into chunks of size in_channels//groups","For groups: split kernels into chunks of size out_channels//groups","Convolve each input group with its corresponding kernel group","Concatenate results along the channel dimension"],solution:`import numpy as np

def conv2d_advanced(input, kernel, padding=0, stride=1, groups=1):
    input = np.array(input, dtype=float)
    kernel = np.array(kernel, dtype=float)

    batch, in_channels, H, W = input.shape
    out_channels, kernel_in_channels, kH, kW = kernel.shape

    # Pad input
    if padding > 0:
        input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    _, _, H_padded, W_padded = input.shape

    # Calculate output dimensions
    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1

    # Initialize output
    output = np.zeros((batch, out_channels, H_out, W_out))

    # Channels per group
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    for g in range(groups):
        # Input channels for this group
        in_start = g * in_channels_per_group
        in_end = in_start + in_channels_per_group

        # Output channels for this group
        out_start = g * out_channels_per_group
        out_end = out_start + out_channels_per_group

        # Get the input slice for this group
        input_group = input[:, in_start:in_end, :, :]

        # Get the kernels for this group
        kernel_group = kernel[out_start:out_end, :, :, :]

        # Perform convolution for this group
        for b in range(batch):
            for oc in range(out_channels_per_group):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        region = input_group[b, :, h_start:h_start+kH, w_start:w_start+kW]
                        output[b, out_start + oc, i, j] = np.sum(region * kernel_group[oc])

    return output
`}],db=[],cb=[{id:"scaled-dot-product-attention",title:"Scaled Dot-Product Attention",section:"transformers",difficulty:"medium",description:`
## Scaled Dot-Product Attention

Implement the core attention mechanism from "Attention Is All You Need".

### Formula
\`\`\`
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
\`\`\`

Where:
- Q: Queries (seq_len, d_k)
- K: Keys (seq_len, d_k)
- V: Values (seq_len, d_v)
- d_k: Dimension of keys

### Steps
1. Compute attention scores: Q @ K.T
2. Scale by sqrt(d_k)
3. Apply softmax
4. Multiply by V
    `,examples:[{input:"Q, K, V all (4, 8) - 4 tokens, 8 dims",output:"output (4, 8)",explanation:"Each token attends to all tokens"}],starterCode:`import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Queries (seq_len, d_k)
        K: Keys (seq_len, d_k)
        V: Values (seq_len, d_v)

    Returns:
        output: Attention output (seq_len, d_v)
        attention_weights: Attention weights (seq_len, seq_len)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape correct",input:"scaled_dot_product_attention(np.random.randn(4, 8), np.random.randn(4, 8), np.random.randn(4, 8))[0].shape",expected:"(4, 8)",hidden:!1},{id:"2",description:"Attention weights sum to 1",input:"bool(np.allclose(scaled_dot_product_attention(np.ones((4, 8)), np.ones((4, 8)), np.ones((4, 8)))[1].sum(axis=-1), 1.0))",expected:"True",hidden:!1},{id:"3",description:"Matches a reference implementation",input:"(lambda Q, K, V: (lambda out, w: bool(np.allclose(w, (lambda s: np.exp(s - s.max(-1, keepdims=True)) / np.exp(s - s.max(-1, keepdims=True)).sum(-1, keepdims=True))(Q @ K.T / np.sqrt(4))) and np.allclose(out, w @ V)))(*scaled_dot_product_attention(Q, K, V)))(np.random.randn(3, 4), np.random.randn(5, 4), np.random.randn(5, 2))",expected:"True",hidden:!0}],hints:["Compute scores = Q @ K.T","Scale by sqrt(d_k) where d_k = Q.shape[-1]","Apply softmax along the last axis","Return softmax(scores) @ V"],solution:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)

    # Apply softmax
    attention_weights = softmax(scores)

    # Compute output
    output = attention_weights @ V

    return output, attention_weights
`},{id:"multi-head-attention",title:"Multi-Head Attention",section:"transformers",difficulty:"hard",description:`
## Multi-Head Attention

Implement multi-head attention from the Transformer architecture.

### Concept
\`\`\`
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
\`\`\`

### Steps
1. Project Q, K, V with different learned projections for each head
2. Apply scaled dot-product attention in parallel
3. Concatenate heads
4. Project back to original dimension
    `,examples:[{input:"X (4, 64), num_heads=8, d_model=64",output:"output (4, 64)",explanation:"8 heads with d_k=8 each, concatenated to 64"}],starterCode:`import numpy as np

def multi_head_attention(Q, K, V, num_heads):
    """
    Multi-head attention mechanism.

    Args:
        Q: Queries (seq_len, d_model)
        K: Keys (seq_len, d_model)
        V: Values (seq_len, d_model)
        num_heads: Number of attention heads

    Returns:
        output: Multi-head attention output (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape preserved",input:"multi_head_attention(np.random.randn(4, 64), np.random.randn(4, 64), np.random.randn(4, 64), 8).shape",expected:"(4, 64)",hidden:!1},{id:"2",description:"One head equals plain attention",input:"(lambda Q, K, V: bool(np.allclose(multi_head_attention(Q, K, V, 1), (lambda s: (np.exp(s - s.max(-1, keepdims=True)) / np.exp(s - s.max(-1, keepdims=True)).sum(-1, keepdims=True)) @ V)(Q @ K.T / np.sqrt(8)))))(np.random.randn(4, 8), np.random.randn(4, 8), np.random.randn(4, 8))",expected:"True",hidden:!0}],hints:["Reshape Q, K, V to (seq_len, num_heads, d_k)","Apply attention for each head","Reshape back to (seq_len, d_model)"],solution:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def multi_head_attention(Q, K, V, num_heads):
    seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Reshape to (seq_len, num_heads, d_k)
    Q = Q.reshape(seq_len, num_heads, d_k)
    K = K.reshape(seq_len, num_heads, d_k)
    V = V.reshape(seq_len, num_heads, d_k)

    # Transpose to (num_heads, seq_len, d_k)
    Q = Q.transpose(1, 0, 2)
    K = K.transpose(1, 0, 2)
    V = V.transpose(1, 0, 2)

    # Scaled dot-product attention for each head
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    attention = softmax(scores)
    heads = attention @ V

    # Transpose and reshape back
    heads = heads.transpose(1, 0, 2)  # (seq_len, num_heads, d_k)
    output = heads.reshape(seq_len, d_model)

    return output
`},{id:"positional-encoding",title:"Sinusoidal Positional Encoding",section:"transformers",difficulty:"medium",description:`
## Sinusoidal Positional Encoding

Implement the positional encoding from the original Transformer paper.

### Formulas
\`\`\`
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
\`\`\`

Where:
- pos: Position in sequence
- i: Dimension index
- d_model: Model dimension

### Purpose
- Injects position information into embeddings
- Allows model to learn relative positions
- Fixed (not learned) encodings
    `,examples:[{input:"max_len=100, d_model=512",output:"PE matrix (100, 512)",explanation:"Position encoding for 100 positions"}],starterCode:`import numpy as np

def positional_encoding(max_len, d_model):
    """
    Generate sinusoidal positional encodings.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        PE: Positional encoding matrix (max_len, d_model)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Correct shape",input:"positional_encoding(50, 64).shape",expected:"(50, 64)",hidden:!1},{id:"2",description:"First position starts with sin(0)=0",input:"bool(np.isclose(positional_encoding(10, 4)[0, 0], 0.0))",expected:"True",hidden:!0}],hints:["Create position indices: np.arange(max_len)","Create dimension indices for the divisor","Use div_term = 10000^(2i/d_model)","Apply sin to even indices, cos to odd indices"],solution:`import numpy as np

def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))

    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
`},{id:"layer-norm",title:"Layer Normalization",section:"transformers",difficulty:"easy",description:`
## Layer Normalization

Implement layer normalization, used in Transformers instead of batch normalization.

### Formula
\`\`\`
LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
\`\`\`

### Difference from BatchNorm
- **BatchNorm**: Normalizes across batch dimension
- **LayerNorm**: Normalizes across feature dimension
- LayerNorm works better for sequence models
    `,examples:[{input:"X (batch, seq_len, features)",output:"Normalized X, each position normalized independently",explanation:"Normalize across last dimension"}],starterCode:`import numpy as np

def layer_norm(X, gamma, beta, eps=1e-5):
    """
    Apply layer normalization.

    Args:
        X: Input (batch, seq_len, features) or (seq_len, features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant

    Returns:
        output: Normalized tensor
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output mean near zero",input:"bool(np.allclose(layer_norm(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), np.ones(4), np.zeros(4)).mean(axis=-1), 0.0, atol=1e-5))",expected:"True",hidden:!1},{id:"2",description:"Gamma and beta work",input:"bool(np.allclose(layer_norm(np.array([[0.0, 0.0, 2.0, 2.0]]), np.array([1.0, 2.0, 1.0, 2.0]), np.zeros(4)), np.array([[-1.0, -2.0, 1.0, 2.0]])))",expected:"True",hidden:!0}],hints:["Compute mean and variance along the last axis","Use keepdims=True for proper broadcasting","Apply normalization: (x - mean) / sqrt(var + eps)"],solution:`import numpy as np

def layer_norm(X, gamma, beta, eps=1e-5):
    # Compute statistics along last dimension
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)

    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)

    # Scale and shift
    return gamma * X_norm + beta
`},{id:"causal-mask",title:"Causal Attention Mask",section:"transformers",difficulty:"easy",description:`
## Causal (Autoregressive) Mask

Create a causal mask for decoder self-attention to prevent attending to future tokens.

### Mask Structure
\`\`\`
[[0, -inf, -inf, -inf],
 [0,    0, -inf, -inf],
 [0,    0,    0, -inf],
 [0,    0,    0,    0]]
\`\`\`

### Usage
- Add mask to attention scores before softmax
- -inf becomes 0 after softmax
- Ensures each position only attends to previous positions
    `,examples:[{input:"seq_len=4",output:"Lower triangular mask (4, 4)",explanation:"Position i can only attend to positions <= i"}],starterCode:`import numpy as np

def create_causal_mask(seq_len):
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length

    Returns:
        mask: Causal mask (seq_len, seq_len)
               0 for allowed positions, -inf for masked
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Correct shape and pattern",input:"create_causal_mask(4).shape",expected:"(4, 4)",hidden:!1},{id:"2",description:"Lower triangular zeros, upper triangular -inf",input:'bool(np.allclose(create_causal_mask(3), np.array([[0.0, float("-inf"), float("-inf")], [0.0, 0.0, float("-inf")], [0.0, 0.0, 0.0]])))',expected:"True",hidden:!0}],hints:["Create a matrix of ones using np.ones((seq_len, seq_len))","Use np.triu to get upper triangular matrix (excluding diagonal)","Replace 1s with -inf"],solution:`import numpy as np

def create_causal_mask(seq_len):
    # Create upper triangular matrix (above diagonal)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    # Replace 1s with -inf
    mask = np.where(mask == 1, float('-inf'), 0.0)
    return mask
`}],pb=[],hb=[],mb=[{id:"vae-reparameterization",title:"VAE Reparameterization Trick",section:"generative-models",difficulty:"medium",description:`
## VAE Reparameterization Trick

Implement the reparameterization trick that allows backpropagation through stochastic sampling in VAEs.

### The Problem
We want to sample z ~ N(μ, σ²), but sampling is not differentiable.

### The Solution
\`\`\`
ε ~ N(0, 1)
z = μ + σ * ε
\`\`\`

### Why It Works
- ε is sampled independently of parameters
- z is now a deterministic function of μ, σ, and ε
- Gradients can flow through μ and σ
    `,examples:[{input:"mu = [0, 1], log_var = [0, 0]",output:"z = mu + exp(0.5 * log_var) * epsilon",explanation:"log_var=0 means σ=1"}],starterCode:`import numpy as np

def reparameterize(mu, log_var):
    """
    Sample from latent distribution using reparameterization trick.

    Args:
        mu: Mean of latent distribution (batch, latent_dim)
        log_var: Log variance of latent distribution (batch, latent_dim)

    Returns:
        z: Sampled latent vectors (batch, latent_dim)
    """
    np.random.seed(42)  # For reproducibility
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape matches input",input:"reparameterize(np.zeros((2, 4)), np.zeros((2, 4))).shape",expected:"(2, 4)",hidden:!1},{id:"2",description:"Zero variance case has correct shape",input:"reparameterize(np.ones((3, 5)), np.full((3, 5), -1000)).shape",expected:"(3, 5)",hidden:!0}],hints:["std = exp(0.5 * log_var)","Sample epsilon from standard normal","z = mu + std * epsilon"],solution:`import numpy as np

def reparameterize(mu, log_var):
    np.random.seed(42)
    # Compute standard deviation
    std = np.exp(0.5 * log_var)
    # Sample epsilon from standard normal
    eps = np.random.randn(*mu.shape)
    # Reparameterization trick
    z = mu + std * eps
    return z
`},{id:"vae-loss",title:"VAE Loss Function",section:"generative-models",difficulty:"medium",description:`
## VAE Loss (ELBO)

Implement the VAE loss function: reconstruction loss + KL divergence.

### Loss Components
\`\`\`
L = L_reconstruction + L_KL

L_reconstruction = MSE(x, x_reconstructed)
L_KL = -0.5 * sum(1 + log_var - mu² - exp(log_var))   # sum over latent dims, mean over batch
\`\`\`

### Intuition
- **Reconstruction loss**: Output should match input
- **KL divergence**: Latent distribution should be close to N(0,1)
    `,examples:[{input:"Perfect reconstruction, mu=0, var=1",output:"Loss ≈ 0",explanation:"Both terms are minimized"}],starterCode:`import numpy as np

def vae_loss(x, x_reconstructed, mu, log_var):
    """
    Compute VAE loss (negative ELBO).

    Args:
        x: Original input (batch, features)
        x_reconstructed: Reconstructed input (batch, features)
        mu: Latent mean (batch, latent_dim)
        log_var: Latent log variance (batch, latent_dim)

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Perfect case",input:"(np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 2)), np.zeros((2, 2)))",expected:"(0.0, 0.0, 0.0)",hidden:!1},{id:"2",description:"Non-zero KL",input:"(np.zeros((1, 4)), np.zeros((1, 4)), np.ones((1, 2)), np.zeros((1, 2)))",expected:"(1.0, 0.0, 1.0)",hidden:!0}],hints:["Reconstruction: np.mean((x - x_reconstructed)²)","KL: -0.5 * sum(1 + log_var - mu² - exp(log_var))","Average over batch"],solution:`import numpy as np

def vae_loss(x, x_reconstructed, mu, log_var):
    # Reconstruction loss (MSE)
    recon_loss = np.mean((x - x_reconstructed) ** 2)

    # KL divergence
    kl_loss = -0.5 * np.mean(np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1))

    total_loss = recon_loss + kl_loss

    return round(total_loss, 4), round(recon_loss, 4), round(kl_loss, 4)
`},{id:"diffusion-noise-schedule",title:"Diffusion Noise Schedule",section:"generative-models",difficulty:"easy",description:`
## Diffusion Noise Schedule

Implement a linear noise schedule for diffusion models.

### Linear Schedule
\`\`\`
β_t = β_start + t * (β_end - β_start) / (T - 1)

where t = 0, 1, ..., T-1
\`\`\`

### Derived Quantities
\`\`\`
α_t = 1 - β_t
ᾱ_t = prod(α_1, ..., α_t)  # cumulative product
\`\`\`

These control how much noise is added at each step.
    `,examples:[{input:"T=1000, beta_start=0.0001, beta_end=0.02",output:"betas, alphas, alpha_bars arrays",explanation:"Standard DDPM schedule"}],starterCode:`import numpy as np

def linear_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    """
    Create linear noise schedule for diffusion.

    Args:
        T: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        betas: Beta values (T,)
        alphas: Alpha values (T,)
        alpha_bars: Cumulative alpha product (T,)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Beta range correct",input:"linear_noise_schedule(100, 0.0001, 0.02)[0].shape",expected:"(100,)",hidden:!1},{id:"2",description:"Alpha bar decreases - last < first",input:"bool(linear_noise_schedule(50, 0.001, 0.01)[2][-1] < linear_noise_schedule(50, 0.001, 0.01)[2][0])",expected:"True",hidden:!0}],hints:["Use np.linspace for linear interpolation","alphas = 1 - betas","alpha_bars = np.cumprod(alphas)"],solution:`import numpy as np

def linear_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    # Linear schedule
    betas = np.linspace(beta_start, beta_end, T)

    # Compute alphas
    alphas = 1 - betas

    # Cumulative product
    alpha_bars = np.cumprod(alphas)

    return betas, alphas, alpha_bars
`},{id:"diffusion-forward",title:"Diffusion Forward Process",section:"generative-models",difficulty:"medium",description:`
## Diffusion Forward Process (Adding Noise)

Implement the forward diffusion process that adds noise to data.

### Formula
\`\`\`
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

where ε ~ N(0, I)
\`\`\`

### Intuition
- As t increases, ᾱ_t decreases
- More noise is added, signal is reduced
- At t=T, x_T ≈ pure noise
    `,examples:[{input:"x_0 (image), t=500, T=1000",output:"x_t (noisy image)",explanation:"Halfway through diffusion process"}],starterCode:`import numpy as np

def diffusion_forward(x_0, t, alpha_bars):
    """
    Add noise to data using forward diffusion.

    Args:
        x_0: Original data (batch, ...)
        t: Timestep (int)
        alpha_bars: Cumulative alpha products (T,)

    Returns:
        x_t: Noisy data at timestep t
        noise: The noise that was added
    """
    np.random.seed(42)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape matches input",input:"diffusion_forward(np.ones((2, 4)), 50, np.linspace(0.99, 0.01, 100))[0].shape",expected:"(2, 4)",hidden:!1},{id:"2",description:"Noise shape matches input",input:"diffusion_forward(np.ones((3, 5)), 10, np.linspace(0.99, 0.01, 100))[1].shape",expected:"(3, 5)",hidden:!0}],hints:["Get alpha_bar_t = alpha_bars[t]","Sample noise from standard normal","Apply formula: sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise"],solution:`import numpy as np

def diffusion_forward(x_0, t, alpha_bars):
    np.random.seed(42)

    alpha_bar_t = alpha_bars[t]

    # Sample noise
    noise = np.random.randn(*x_0.shape)

    # Forward process
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * noise

    return x_t, noise
`},{id:"vqvae-quantization",title:"VQ-VAE Vector Quantization",section:"generative-models",difficulty:"hard",description:`
## VQ-VAE Vector Quantization

Implement the vector quantization layer used in VQ-VAE (Vector Quantized Variational Autoencoder).

### Overview
Unlike standard VAEs that use continuous latent spaces, VQ-VAE uses **discrete** latent representations by mapping encoder outputs to the nearest embedding in a learned codebook.

### Quantization Process
1. **Encoder output**: z_e of shape (batch, H, W, D)
2. **Codebook**: K embedding vectors of dimension D
3. **Find nearest**: For each spatial position, find the closest codebook entry
4. **Quantize**: Replace z_e with the nearest codebook vector z_q

### Distance Calculation
\`\`\`
distances[b, h, w, k] = ||z_e[b, h, w] - codebook[k]||²
\`\`\`

### Straight-Through Estimator
During training, gradients flow through z_q to z_e by copying gradients:
\`\`\`
z_q = z_e + stop_gradient(z_q - z_e)
\`\`\`
(We don't implement this here, just the forward pass)

### VQ-VAE Loss Components
\`\`\`
L = reconstruction + ||sg[z_e] - e||² + β * ||z_e - sg[e]||²
\`\`\`
- **Codebook loss**: Moves embeddings toward encoder outputs
- **Commitment loss**: Commits encoder to embeddings

### Function Signature
\`\`\`python
def vq_quantize(z_e, codebook):
    # z_e: (batch, H, W, embedding_dim) - encoder output
    # codebook: (num_embeddings, embedding_dim) - K embedding vectors
    # Returns: z_q, indices
\`\`\`
    `,examples:[{input:"z_e (2, 4, 4, 64), codebook (512, 64)",output:"z_q (2, 4, 4, 64), indices (2, 4, 4)",explanation:"Each spatial position mapped to one of 512 codes"}],starterCode:`import numpy as np

def vq_quantize(z_e, codebook):
    """
    Vector quantization for VQ-VAE.

    Args:
        z_e: Encoder output (batch, H, W, embedding_dim)
        codebook: Embedding vectors (num_embeddings, embedding_dim)

    Returns:
        z_q: Quantized vectors (batch, H, W, embedding_dim)
        indices: Codebook indices used (batch, H, W)
    """
    # Your code here
    pass


def vq_loss(z_e, z_q, beta=0.25):
    """
    Compute VQ-VAE losses (codebook + commitment).

    Args:
        z_e: Encoder output (batch, H, W, embedding_dim)
        z_q: Quantized vectors (batch, H, W, embedding_dim)
        beta: Commitment loss weight (default: 0.25)

    Returns:
        codebook_loss: Loss to update codebook
        commitment_loss: Loss to commit encoder to codebook
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"z_q output shape matches input",input:"vq_quantize(np.random.randn(2, 4, 4, 8), np.random.randn(16, 8))[0].shape",expected:"(2, 4, 4, 8)",hidden:!1},{id:"2",description:"Indices shape is correct",input:"vq_quantize(np.random.randn(2, 4, 4, 8), np.random.randn(16, 8))[1].shape",expected:"(2, 4, 4)",hidden:!1},{id:"3",description:"Indices are valid codebook indices (all less than K)",input:"bool(np.all(vq_quantize(np.random.randn(2, 3, 3, 4), np.random.randn(8, 4))[1] < 8))",expected:"True",hidden:!0},{id:"4",description:"VQ loss returns tuple of two values",input:"len(vq_loss(np.random.randn(2, 3, 3, 4), np.random.randn(2, 3, 3, 4), 0.25))",expected:"2",hidden:!0}],hints:["Reshape z_e to (batch*H*W, D) for easier distance computation","Use broadcasting: ||a - b||² = ||a||² + ||b||² - 2*a·b","np.argmin along the codebook axis gives indices","Index into codebook with the indices to get z_q","Reshape back to original spatial dimensions","For loss: codebook_loss = ||z_e.detach() - z_q||², commitment = ||z_e - z_q.detach()||²"],solution:`import numpy as np

def vq_quantize(z_e, codebook):
    z_e = np.array(z_e)
    codebook = np.array(codebook)

    batch, H, W, D = z_e.shape
    K, _ = codebook.shape

    # Flatten spatial dimensions: (batch*H*W, D)
    z_flat = z_e.reshape(-1, D)

    # Compute distances using: ||a-b||² = ||a||² + ||b||² - 2*a·b
    # z_flat: (N, D), codebook: (K, D)
    z_sq = np.sum(z_flat ** 2, axis=1, keepdims=True)  # (N, 1)
    codebook_sq = np.sum(codebook ** 2, axis=1)        # (K,)
    cross = z_flat @ codebook.T                        # (N, K)

    distances = z_sq + codebook_sq - 2 * cross         # (N, K)

    # Find nearest codebook entry
    indices_flat = np.argmin(distances, axis=1)        # (N,)

    # Get quantized vectors
    z_q_flat = codebook[indices_flat]                  # (N, D)

    # Reshape back
    z_q = z_q_flat.reshape(batch, H, W, D)
    indices = indices_flat.reshape(batch, H, W)

    return z_q, indices


def vq_loss(z_e, z_q, beta=0.25):
    z_e = np.array(z_e)
    z_q = np.array(z_q)

    # Codebook loss: ||sg[z_e] - z_q||² (moves codebook toward encoder output)
    # In practice, sg[z_e] means z_e is treated as constant
    codebook_loss = np.mean((z_e - z_q) ** 2)

    # Commitment loss: ||z_e - sg[z_q]||² (commits encoder to codebook)
    # Same computation, but gradient only flows to z_e
    commitment_loss = beta * np.mean((z_e - z_q) ** 2)

    return round(codebook_loss, 4), round(commitment_loss, 4)
`},{id:"kl-divergence",title:"KL Divergence (Gaussians)",section:"generative-models",difficulty:"easy",description:`
## KL Divergence Between Gaussians

Compute the KL divergence between two univariate Gaussian distributions.

### Formula
\`\`\`
KL(P || Q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²) / (2σ_q²) - 0.5
\`\`\`

Where P = N(μ_p, σ_p²) and Q = N(μ_q, σ_q²)

### Properties
- KL ≥ 0
- KL = 0 iff P = Q
- Not symmetric: KL(P||Q) ≠ KL(Q||P)
    `,examples:[{input:"P = N(0, 1), Q = N(0, 1)",output:"0",explanation:"Same distribution"}],starterCode:`import numpy as np

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute KL divergence between two Gaussians.

    Args:
        mu_p, sigma_p: Mean and std of P
        mu_q, sigma_q: Mean and std of Q

    Returns:
        kl: KL(P || Q)
    """
    # Your code here
    pass
`,testCases:[{id:"1",description:"Same distribution",input:"(0, 1, 0, 1)",expected:"0.0",hidden:!1},{id:"2",description:"Different means",input:"(1, 1, 0, 1)",expected:"0.5",hidden:!1},{id:"3",description:"Different variances",input:"(0, 2, 0, 1)",expected:"0.8069",hidden:!0}],hints:["Apply the formula directly","Use np.log for natural logarithm","Remember: σ² is variance, σ is std"],solution:`import numpy as np

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    term1 = np.log(sigma_q / sigma_p)
    term2 = (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
    term3 = -0.5

    kl = term1 + term2 + term3
    return round(kl, 4)
`}],fb=[{id:"rl-discounted-return",title:"Discounted Return",section:"reinforcement-learning",difficulty:"easy",description:`
## Discounted Return (Cumulative Reward)

Calculate the discounted return (cumulative reward) from a sequence of rewards.

### Formula
\`\`\`
G_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ... + γ^(T-t) * r_T
    = Σ_{k=0}^{T-t} γ^k * r_{t+k}
\`\`\`

Where:
- \`r_t\`: Reward at timestep t
- \`γ\` (gamma): Discount factor (0 ≤ γ ≤ 1)
- \`T\`: Final timestep

### Why Discount?
- Future rewards are uncertain
- Prefer immediate rewards
- γ = 0: Only care about immediate reward
- γ = 1: All rewards equally important
- Typical values: 0.9 to 0.99

### Function Signature
\`\`\`python
def discounted_return(rewards: np.ndarray, gamma: float) -> float:
\`\`\`

Returns the total discounted return from timestep 0.
    `,examples:[{input:"rewards = [1, 1, 1, 1], gamma = 0.9",output:"3.439",explanation:"1 + 0.9*1 + 0.81*1 + 0.729*1 = 3.439"},{input:"rewards = [0, 0, 0, 10], gamma = 0.5",output:"1.25",explanation:"0 + 0 + 0 + 0.125*10 = 1.25"}],starterCode:`import numpy as np

def discounted_return(rewards: np.ndarray, gamma: float) -> float:
    """
    Calculate the discounted cumulative return.

    Args:
        rewards: Array of rewards [r_0, r_1, ..., r_T]
        gamma: Discount factor (0 <= gamma <= 1)

    Returns:
        Total discounted return G_0
    """
    rewards = np.array(rewards)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Constant rewards",input:"([1, 1, 1, 1], 0.9)",expected:"3.439",hidden:!1},{id:"2",description:"Delayed reward",input:"([0, 0, 0, 10], 0.5)",expected:"1.25",hidden:!1},{id:"3",description:"No discount",input:"([1, 2, 3, 4], 1.0)",expected:"10.0",hidden:!1},{id:"4",description:"Only immediate",input:"([5, 10, 15], 0.0)",expected:"5.0",hidden:!0}],hints:["Create an array of discount factors: [γ⁰, γ¹, γ², ...]","Use np.arange to create exponents, then gamma ** exponents","Multiply rewards by discount factors and sum"],solution:`import numpy as np

def discounted_return(rewards: np.ndarray, gamma: float) -> float:
    """
    Calculate the discounted cumulative return.
    """
    rewards = np.array(rewards)
    T = len(rewards)

    # Create discount factors: [gamma^0, gamma^1, gamma^2, ...]
    discounts = gamma ** np.arange(T)

    # Compute discounted return
    G = np.sum(rewards * discounts)

    return round(G, 3)
`},{id:"rl-epsilon-greedy",title:"Epsilon-Greedy Policy",section:"reinforcement-learning",difficulty:"easy",description:`
## Epsilon-Greedy Policy

Implement the epsilon-greedy action selection policy, which balances exploration and exploitation.

### Algorithm
\`\`\`
With probability ε: choose random action (explore)
With probability 1-ε: choose action with highest Q-value (exploit)
\`\`\`

### Why Epsilon-Greedy?
- **Exploration**: Try new actions to discover better strategies
- **Exploitation**: Use known good actions to maximize reward
- ε typically decays over time (start exploring, then exploit)

### Function Signature
\`\`\`python
def epsilon_greedy(q_values: np.ndarray, epsilon: float,
                   random_value: float) -> int:
\`\`\`

**Note**: We pass \`random_value\` (a number between 0-1) instead of generating it,
so tests are deterministic. In practice, you'd use \`np.random.random()\`.

### Expected Return
- If \`random_value < epsilon\`: return \`random_action\` (passed as floor of random_value * n_actions for testing)
- Else: return \`argmax(q_values)\`
    `,examples:[{input:"q_values = [1.0, 3.0, 2.0], epsilon = 0.1, random_value = 0.5",output:"1",explanation:"0.5 >= 0.1, so exploit: argmax([1,3,2]) = 1"},{input:"q_values = [1.0, 3.0, 2.0], epsilon = 0.9, random_value = 0.5",output:"1",explanation:"0.5 < 0.9, so explore: floor(0.5 * 3) = 1"}],starterCode:`import numpy as np

def epsilon_greedy(q_values: np.ndarray, epsilon: float,
                   random_value: float) -> int:
    """
    Select action using epsilon-greedy policy.

    Args:
        q_values: Q-values for each action [Q(a_0), Q(a_1), ...]
        epsilon: Exploration probability (0 <= epsilon <= 1)
        random_value: Pre-generated random number for deterministic testing

    Returns:
        Selected action index
    """
    q_values = np.array(q_values)
    n_actions = len(q_values)

    # Your code here
    # If random_value < epsilon: explore (return floor(random_value * n_actions))
    # Else: exploit (return argmax of q_values)
    pass
`,testCases:[{id:"1",description:"Exploit (greedy)",input:"([1.0, 3.0, 2.0], 0.1, 0.5)",expected:"1",hidden:!1},{id:"2",description:"Explore",input:"([1.0, 3.0, 2.0], 0.9, 0.5)",expected:"1",hidden:!1},{id:"3",description:"Always exploit (epsilon=0)",input:"([5.0, 2.0, 8.0, 1.0], 0.0, 0.99)",expected:"2",hidden:!1},{id:"4",description:"Explore with low random",input:"([1.0, 2.0, 3.0, 4.0], 0.5, 0.2)",expected:"0",hidden:!0}],hints:["Compare random_value to epsilon to decide explore vs exploit","For exploration, compute floor(random_value * n_actions)","For exploitation, use np.argmax(q_values)"],solution:`import numpy as np

def epsilon_greedy(q_values: np.ndarray, epsilon: float,
                   random_value: float) -> int:
    """
    Select action using epsilon-greedy policy.
    """
    q_values = np.array(q_values)
    n_actions = len(q_values)

    if random_value < epsilon:
        # Explore: random action
        return int(random_value * n_actions)
    else:
        # Exploit: greedy action
        return int(np.argmax(q_values))
`},{id:"rl-bellman-value",title:"Bellman Expectation (State Value)",section:"reinforcement-learning",difficulty:"easy",description:`
## Bellman Expectation Equation for State Value

Compute the value of a state given transition probabilities, rewards, and next-state values.

### Bellman Expectation Equation
\`\`\`
V(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V(s')]
\`\`\`

For a fixed policy, this simplifies to:
\`\`\`
V(s) = Σ_{s'} P(s'|s) * [R(s,s') + γ * V(s')]
\`\`\`

### What It Means
The value of state s equals the expected:
- Immediate reward R
- Plus discounted value of next state V(s')

### Function Signature
\`\`\`python
def bellman_value(transitions: list, gamma: float,
                  next_values: np.ndarray) -> float:
\`\`\`

Where each transition is \`(probability, reward, next_state_index)\`.
    `,examples:[{input:"transitions=[(0.5, 1, 0), (0.5, 2, 1)], gamma=0.9, V=[5, 10]",output:"8.25",explanation:"0.5*(1 + 0.9*5) + 0.5*(2 + 0.9*10) = 0.5*5.5 + 0.5*11 = 8.25"}],starterCode:`import numpy as np

def bellman_value(transitions: list, gamma: float,
                  next_values: np.ndarray) -> float:
    """
    Compute state value using Bellman expectation equation.

    Args:
        transitions: List of (probability, reward, next_state_index) tuples
        gamma: Discount factor
        next_values: Array of values V(s') for each state

    Returns:
        Value V(s) for the current state
    """
    next_values = np.array(next_values)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Two transitions",input:"bellman_value([(0.5, 1, 0), (0.5, 2, 1)], 0.9, [5, 10])",expected:"8.25",hidden:!1},{id:"2",description:"Deterministic transition",input:"bellman_value([(1.0, 5, 0)], 0.9, [10])",expected:"14.0",hidden:!1},{id:"3",description:"Three outcomes",input:"bellman_value([(0.2, 0, 0), (0.5, 1, 1), (0.3, 2, 2)], 0.9, [0, 5, 10])",expected:"6.05",hidden:!0}],hints:["Loop through each transition (p, r, s_next)","For each: add p * (r + gamma * V[s_next]) to the total","This is the expected value over all possible transitions"],solution:`import numpy as np

def bellman_value(transitions: list, gamma: float,
                  next_values: np.ndarray) -> float:
    """
    Compute state value using Bellman expectation equation.
    """
    next_values = np.array(next_values)

    value = 0.0
    for prob, reward, next_state in transitions:
        # V(s) = Σ P(s'|s) * [R + γ * V(s')]
        value += prob * (reward + gamma * next_values[next_state])

    return round(value, 2)
`},{id:"rl-q-learning-update",title:"Q-Learning Update",section:"reinforcement-learning",difficulty:"medium",description:`
## Q-Learning Update Rule

Implement the Q-learning update rule, the core of this model-free RL algorithm.

### Q-Learning Update
\`\`\`
Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
\`\`\`

Where:
- \`α\`: Learning rate
- \`r\`: Reward received
- \`γ\`: Discount factor
- \`s'\`: Next state
- \`max_a' Q(s', a')\`: Best Q-value in next state

### Key Properties
- **Off-policy**: Updates use max Q-value, not the action actually taken
- **Model-free**: Doesn't need environment dynamics
- **Converges**: To optimal Q* under certain conditions

### Function Signature
\`\`\`python
def q_learning_update(Q: np.ndarray, state: int, action: int,
                      reward: float, next_state: int,
                      alpha: float, gamma: float) -> np.ndarray:
\`\`\`

Returns the updated Q-table.
    `,examples:[{input:"Q=[[0,0],[0,0]], s=0, a=1, r=1, s'=1, α=0.1, γ=0.9",output:"Q[0,1] = 0.1",explanation:"Q[0,1] += 0.1 * (1 + 0.9*0 - 0) = 0.1"}],starterCode:`import numpy as np

def q_learning_update(Q: np.ndarray, state: int, action: int,
                      reward: float, next_state: int,
                      alpha: float, gamma: float) -> np.ndarray:
    """
    Perform one Q-learning update.

    Args:
        Q: Q-table of shape (n_states, n_actions)
        state: Current state index
        action: Action taken
        reward: Reward received
        next_state: Next state index
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        Updated Q-table
    """
    Q = np.array(Q, dtype=float)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Simple update",input:"([[0, 0], [0, 0]], 0, 1, 1.0, 1, 0.1, 0.9)",expected:"[[0.0, 0.1], [0.0, 0.0]]",hidden:!1},{id:"2",description:"With existing values",input:"([[1, 2], [3, 4]], 0, 0, 5.0, 1, 0.5, 0.9)",expected:"[[4.8, 2.0], [3.0, 4.0]]",hidden:!1},{id:"3",description:"Terminal state (no future)",input:"([[0, 0], [0, 0]], 0, 0, 10.0, 1, 1.0, 0.0)",expected:"[[10.0, 0.0], [0.0, 0.0]]",hidden:!0}],hints:["Find the maximum Q-value in the next state: np.max(Q[next_state])","Compute TD target: reward + gamma * max_next_Q","Compute TD error: target - Q[state, action]","Update: Q[state, action] += alpha * td_error"],solution:`import numpy as np

def q_learning_update(Q: np.ndarray, state: int, action: int,
                      reward: float, next_state: int,
                      alpha: float, gamma: float) -> np.ndarray:
    """
    Perform one Q-learning update.
    """
    Q = np.array(Q, dtype=float)

    # Best Q-value in next state
    max_next_Q = np.max(Q[next_state])

    # TD target (Bellman optimality target)
    td_target = reward + gamma * max_next_Q

    # TD error
    td_error = td_target - Q[state, action]

    # Update Q-value
    Q[state, action] += alpha * td_error

    return np.round(Q, 1).tolist()
`},{id:"rl-sarsa-update",title:"SARSA Update",section:"reinforcement-learning",difficulty:"medium",description:`
## SARSA Update Rule

Implement SARSA (State-Action-Reward-State-Action), an on-policy TD control algorithm.

### SARSA Update
\`\`\`
Q(s, a) ← Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
\`\`\`

### SARSA vs Q-Learning
| Aspect | SARSA | Q-Learning |
|--------|-------|------------|
| Policy | On-policy | Off-policy |
| Next Q | Q(s', a') actual | max_a' Q(s', a') |
| Safety | More conservative | Can be risky |

### On-Policy Meaning
SARSA uses the **actual next action** a' that will be taken (following the current policy),
while Q-learning uses the **best possible** next action.

### Function Signature
\`\`\`python
def sarsa_update(Q: np.ndarray, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float) -> np.ndarray:
\`\`\`
    `,examples:[{input:"Q=[[1,2],[3,4]], s=0, a=0, r=5, s'=1, a'=0, α=0.5, γ=0.9",output:"Q[0,0] = 4.35",explanation:"Q[0,0] += 0.5*(5 + 0.9*3 - 1) = 0.5*6.7 = 3.35, so 1+3.35=4.35"}],starterCode:`import numpy as np

def sarsa_update(Q: np.ndarray, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float) -> np.ndarray:
    """
    Perform one SARSA update.

    Args:
        Q: Q-table of shape (n_states, n_actions)
        state: Current state s
        action: Action taken a
        reward: Reward received r
        next_state: Next state s'
        next_action: Next action a' (will be taken)
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        Updated Q-table
    """
    Q = np.array(Q, dtype=float)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Basic SARSA update",input:"([[1, 2], [3, 4]], 0, 0, 5.0, 1, 0, 0.5, 0.9)",expected:"[[4.35, 2.0], [3.0, 4.0]]",hidden:!1},{id:"2",description:"Different next action",input:"([[1, 2], [3, 4]], 0, 0, 5.0, 1, 1, 0.5, 0.9)",expected:"[[4.8, 2.0], [3.0, 4.0]]",hidden:!1},{id:"3",description:"From zero",input:"([[0, 0], [0, 0]], 0, 1, 1.0, 1, 0, 0.1, 0.9)",expected:"[[0.0, 0.1], [0.0, 0.0]]",hidden:!0}],hints:["The key difference from Q-learning: use Q[next_state, next_action] not max","TD target: reward + gamma * Q[next_state, next_action]","TD error: target - Q[state, action]","Update: Q[state, action] += alpha * td_error"],solution:`import numpy as np

def sarsa_update(Q: np.ndarray, state: int, action: int, reward: float,
                 next_state: int, next_action: int,
                 alpha: float, gamma: float) -> np.ndarray:
    """
    Perform one SARSA update.
    """
    Q = np.array(Q, dtype=float)

    # SARSA uses actual next action, not max
    next_Q = Q[next_state, next_action]

    # TD target
    td_target = reward + gamma * next_Q

    # TD error
    td_error = td_target - Q[state, action]

    # Update Q-value
    Q[state, action] += alpha * td_error

    return np.round(Q, 2).tolist()
`},{id:"rl-td0-prediction",title:"TD(0) Value Prediction",section:"reinforcement-learning",difficulty:"medium",description:`
## TD(0) Value Prediction

Implement TD(0), the simplest temporal difference learning algorithm for value estimation.

### TD(0) Update
\`\`\`
V(s) ← V(s) + α * [r + γ * V(s') - V(s)]
\`\`\`

### Key Concepts
- **TD Error (δ)**: \`r + γ * V(s') - V(s)\`
- **Bootstrap**: Uses estimated V(s') rather than waiting for episode end
- **Online**: Updates after each step, not at episode end

### TD vs Monte Carlo
| Aspect | TD(0) | Monte Carlo |
|--------|-------|-------------|
| Update | Every step | End of episode |
| Bias | Some bias | Unbiased |
| Variance | Low variance | High variance |
| Bootstrap | Yes | No |

### Function Signature
\`\`\`python
def td0_update(V: np.ndarray, state: int, reward: float,
               next_state: int, alpha: float, gamma: float,
               done: bool = False) -> np.ndarray:
\`\`\`

If \`done=True\`, the next state is terminal (V(s')=0).
    `,examples:[{input:"V=[0,0,0], s=0, r=1, s'=1, α=0.1, γ=0.9, done=False",output:"V=[0.1, 0, 0]",explanation:"V[0] += 0.1 * (1 + 0.9*0 - 0) = 0.1"}],starterCode:`import numpy as np

def td0_update(V: np.ndarray, state: int, reward: float,
               next_state: int, alpha: float, gamma: float,
               done: bool = False) -> np.ndarray:
    """
    Perform one TD(0) value update.

    Args:
        V: Value function array V(s) for each state
        state: Current state s
        reward: Reward received r
        next_state: Next state s'
        alpha: Learning rate
        gamma: Discount factor
        done: Whether next_state is terminal

    Returns:
        Updated value function
    """
    V = np.array(V, dtype=float)
    # Your code here
    pass
`,testCases:[{id:"1",description:"Non-terminal transition",input:"([0, 0, 0], 0, 1.0, 1, 0.1, 0.9, False)",expected:"[0.1, 0.0, 0.0]",hidden:!1},{id:"2",description:"Terminal state",input:"([0, 0, 0], 0, 10.0, 1, 0.5, 0.9, True)",expected:"[5.0, 0.0, 0.0]",hidden:!1},{id:"3",description:"With existing values",input:"([5, 10, 15], 1, 2.0, 2, 0.1, 0.9, False)",expected:"[5.0, 10.55, 15.0]",hidden:!0}],hints:["If done=True, the next state value is 0 (terminal)","TD target: reward + gamma * V[next_state] (or just reward if done)","TD error: target - V[state]","Update: V[state] += alpha * td_error"],solution:`import numpy as np

def td0_update(V: np.ndarray, state: int, reward: float,
               next_state: int, alpha: float, gamma: float,
               done: bool = False) -> np.ndarray:
    """
    Perform one TD(0) value update.
    """
    V = np.array(V, dtype=float)

    # Next state value (0 if terminal)
    next_value = 0.0 if done else V[next_state]

    # TD target
    td_target = reward + gamma * next_value

    # TD error
    td_error = td_target - V[state]

    # Update value
    V[state] += alpha * td_error

    return np.round(V, 2).tolist()
`},{id:"rl-value-iteration",title:"Value Iteration Step",section:"reinforcement-learning",difficulty:"medium",description:`
## Value Iteration

Implement one step of the value iteration algorithm for solving MDPs.

### Bellman Optimality Update
\`\`\`
V(s) = max_a Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V(s')]
\`\`\`

For each state, we find the action that maximizes expected value.

### Algorithm
1. For each state s:
   - For each action a, compute Q(s,a)
   - V(s) = max over all Q(s,a)
2. Repeat until convergence

### Function Signature
\`\`\`python
def value_iteration_step(V: np.ndarray, transitions: dict,
                         gamma: float) -> np.ndarray:
\`\`\`

Where \`transitions[s][a]\` is a list of \`(prob, reward, next_state)\` tuples.
    `,examples:[{input:"V=[0,0], transitions for 2-state MDP, gamma=0.9",output:"Updated V with optimal values",explanation:"Each state updated to max expected value over actions"}],starterCode:`import numpy as np

def value_iteration_step(V: np.ndarray, transitions: dict,
                         gamma: float) -> np.ndarray:
    """
    Perform one step of value iteration.

    Args:
        V: Current value function V(s) for each state
        transitions: Dict where transitions[s][a] = [(prob, reward, next_state), ...]
        gamma: Discount factor

    Returns:
        Updated value function
    """
    V = np.array(V, dtype=float)
    n_states = len(V)
    V_new = np.zeros(n_states)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Simple 2-state MDP",input:"([0.0, 0.0], {0: {0: [(1.0, 1.0, 0)], 1: [(1.0, 5.0, 1)]}, 1: {0: [(1.0, 2.0, 0)], 1: [(1.0, 0.0, 1)]}}, 0.9)",expected:"[5.0, 2.0]",hidden:!1},{id:"2",description:"With existing values",input:"([5.0, 2.0], {0: {0: [(1.0, 1.0, 0)], 1: [(1.0, 5.0, 1)]}, 1: {0: [(1.0, 2.0, 0)], 1: [(1.0, 0.0, 1)]}}, 0.9)",expected:"[6.8, 6.5]",hidden:!1},{id:"3",description:"Discounting with a self-loop",input:"([1.0, 0.0], {0: {0: [(1.0, 0.0, 0)], 1: [(0.5, 2.0, 1), (0.5, 0.0, 0)]}, 1: {0: [(1.0, 1.0, 1)]}}, 0.5)",expected:"[1.2, 1.0]",hidden:!0}],hints:["For each state, compute the value of each action","Action value: sum over transitions of prob * (reward + gamma * V[next])","State value: max over all action values","Use the new values, not the old ones being updated"],solution:`import numpy as np

def value_iteration_step(V: np.ndarray, transitions: dict,
                         gamma: float) -> np.ndarray:
    """
    Perform one step of value iteration.
    """
    V = np.array(V, dtype=float)
    n_states = len(V)
    V_new = np.zeros(n_states)

    for s in range(n_states):
        if s not in transitions:
            continue

        # Compute Q-value for each action
        q_values = []
        for a in transitions[s]:
            q_sa = 0.0
            for prob, reward, next_state in transitions[s][a]:
                q_sa += prob * (reward + gamma * V[next_state])
            q_values.append(q_sa)

        # V(s) = max Q(s, a)
        V_new[s] = max(q_values) if q_values else 0.0

    return np.round(V_new, 1).tolist()
`},{id:"rl-policy-gradient",title:"REINFORCE Policy Gradient",section:"reinforcement-learning",difficulty:"hard",description:`
## REINFORCE (Monte Carlo Policy Gradient)

Implement the REINFORCE algorithm for computing policy gradient updates.

### Policy Gradient Theorem
\`\`\`
∇J(θ) = E[Σ_t ∇log π(a_t|s_t; θ) * G_t]
\`\`\`

### REINFORCE Algorithm
1. Collect episode: (s_0, a_0, r_1, s_1, a_1, r_2, ...)
2. For each timestep t:
   - Compute return G_t (discounted sum of future rewards)
   - Compute gradient: ∇log π(a_t|s_t) * G_t
3. Update policy parameters

### Softmax Policy
\`\`\`
π(a|s) = exp(θ[s,a]) / Σ_a' exp(θ[s,a'])
\`\`\`

### Gradient of Log-Softmax
\`\`\`
∇_θ log π(a|s) = one_hot(a) - π(·|s)
\`\`\`
(1 for action taken, minus probability of each action)

### Function Signature
\`\`\`python
def reinforce_gradient(theta: np.ndarray, episode: list,
                       gamma: float) -> np.ndarray:
\`\`\`

Episode is list of (state, action, reward) tuples.

### Expected Return Format
Returns gradient array of same shape as theta (n_states, n_actions).
    `,examples:[{input:"theta (3x2), episode=[(0,1,1), (1,0,2)], gamma=0.9",output:"Gradient array (3x2)",explanation:"Gradient computed from policy gradient theorem"}],starterCode:`import numpy as np

def softmax(x):
    """Compute softmax probabilities."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def reinforce_gradient(theta: np.ndarray, episode: list,
                       gamma: float) -> np.ndarray:
    """
    Compute REINFORCE policy gradient.

    Args:
        theta: Policy parameters (n_states, n_actions)
        episode: List of (state, action, reward) tuples
        gamma: Discount factor

    Returns:
        Gradient of same shape as theta
    """
    theta = np.array(theta, dtype=float)
    n_states, n_actions = theta.shape
    gradient = np.zeros_like(theta)

    # Your code here
    # 1. Compute returns G_t for each timestep
    # 2. For each (s, a, r), compute gradient contribution
    # 3. Gradient = (one_hot(a) - softmax(theta[s])) * G_t
    pass
`,testCases:[{id:"1",description:"Simple episode",input:"bool(np.allclose(reinforce_gradient(np.zeros((2, 2)), [(0, 1, 1.0), (1, 0, 1.0)], 0.9), [[-0.95, 0.95], [0.5, -0.5]], atol=0.01))",expected:"True",hidden:!1},{id:"2",description:"Single step episode",input:"bool(np.allclose(reinforce_gradient(np.zeros((2, 2)), [(0, 0, 5.0)], 0.9), [[2.5, -2.5], [0.0, 0.0]], atol=0.01))",expected:"True",hidden:!1},{id:"3",description:"Gradient shape and direction for a single step",input:"(lambda g: (g.shape, bool(g[1, 1] > 0 and g[1, 0] < 0)))(reinforce_gradient(np.zeros((2, 2)), [(1, 1, 2.0)], 0.9))",expected:"((2, 2), True)",hidden:!0}],hints:["First compute all returns: G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...","For softmax policy: gradient = one_hot(action) - π(·|state)","Multiply gradient by return G_t and accumulate","Use backward iteration to compute returns efficiently"],solution:`import numpy as np

def softmax(x):
    """Compute softmax probabilities."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def reinforce_gradient(theta: np.ndarray, episode: list,
                       gamma: float) -> np.ndarray:
    """
    Compute REINFORCE policy gradient.
    """
    theta = np.array(theta, dtype=float)
    n_states, n_actions = theta.shape
    gradient = np.zeros_like(theta)

    T = len(episode)
    if T == 0:
        return gradient

    # Compute returns for each timestep (backward)
    returns = np.zeros(T)
    G = 0
    for t in range(T - 1, -1, -1):
        _, _, reward = episode[t]
        G = reward + gamma * G
        returns[t] = G

    # Compute gradient
    for t in range(T):
        state, action, _ = episode[t]
        G_t = returns[t]

        # Policy probabilities for this state
        probs = softmax(theta[state])

        # Gradient of log π(a|s) = one_hot(a) - π(·|s)
        grad_log_pi = -probs.copy()
        grad_log_pi[action] += 1.0

        # Accumulate: ∇log π(a|s) * G_t
        gradient[state] += grad_log_pi * G_t

    return np.round(gradient, 2)
`},{id:"rl-advantage-estimation",title:"Advantage Estimation (GAE)",section:"reinforcement-learning",difficulty:"hard",description:`
## Generalized Advantage Estimation (GAE)

Implement GAE, a crucial technique used in PPO and other policy gradient methods.

### Why Advantage?
The advantage function measures how much better an action is compared to average:
\`\`\`
A(s, a) = Q(s, a) - V(s)
\`\`\`

### GAE Formula
\`\`\`
Â_t = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}
\`\`\`

Where δ_t is the TD error:
\`\`\`
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
\`\`\`

### Lambda (λ) Trade-off
- λ = 0: Uses only one-step TD (low variance, high bias)
- λ = 1: Uses full Monte Carlo return (high variance, low bias)
- Typical: λ = 0.95

### Function Signature
\`\`\`python
def compute_gae(rewards: np.ndarray, values: np.ndarray,
                gamma: float, lam: float) -> np.ndarray:
\`\`\`

Where values has length T+1 (includes V(s_T+1), use 0 if terminal).
    `,examples:[{input:"rewards=[1,1,1], values=[0,0,0,0], gamma=0.99, lambda=0.95",output:"advantages=[2.83, 1.94, 1.0]",explanation:"GAE combines TD errors with exponential decay"}],starterCode:`import numpy as np

def compute_gae(rewards: np.ndarray, values: np.ndarray,
                gamma: float, lam: float) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: Rewards r_0, r_1, ..., r_{T-1} (length T)
        values: Values V(s_0), V(s_1), ..., V(s_T) (length T+1)
                Last value is V(s_T) or 0 if terminal
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        Advantages Â_0, Â_1, ..., Â_{T-1} (length T)
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    T = len(rewards)
    advantages = np.zeros(T)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Zero values baseline",input:"bool(np.allclose(compute_gae([1, 1, 1], [0, 0, 0, 0], 0.99, 0.95), [2.825, 1.9405, 1.0], atol=0.01))",expected:"True",hidden:!1},{id:"2",description:"With value estimates",input:"bool(np.allclose(compute_gae([1, 2, 3], [1, 2, 3, 0], 0.9, 0.9), [3.99, 2.7, 0.0], atol=0.1))",expected:"True",hidden:!1},{id:"3",description:"Single step",input:"bool(np.allclose(compute_gae([5], [1, 0], 0.99, 0.95), [4.0], atol=0.01))",expected:"True",hidden:!0}],hints:["Compute TD errors first: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)","Work backwards from the last timestep","GAE formula: Â_t = δ_t + γλ*Â_{t+1}","At the last timestep: Â_{T-1} = δ_{T-1}"],solution:`import numpy as np

def compute_gae(rewards: np.ndarray, values: np.ndarray,
                gamma: float, lam: float) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation.
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    T = len(rewards)
    advantages = np.zeros(T)

    # Compute advantages backwards
    gae = 0.0
    for t in range(T - 1, -1, -1):
        # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] - values[t]

        # GAE: Â_t = δ_t + γλ*Â_{t+1}
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    return np.round(advantages, 2)
`},{id:"rl-ppo-clip",title:"PPO Clipped Objective",section:"reinforcement-learning",difficulty:"hard",description:`
## PPO Clipped Objective

Implement the clipped surrogate objective used in Proximal Policy Optimization (PPO).

### PPO Objective
\`\`\`
L^{CLIP}(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
\`\`\`

Where:
- r_t(θ) = π_θ(a|s) / π_θ_old(a|s) is the probability ratio
- A_t is the advantage estimate
- ε is the clipping parameter (typically 0.2)

### Why Clipping?
- Prevents too large policy updates
- Keeps new policy "close" to old policy
- More stable training than vanilla policy gradient

### Key Insight
- If A > 0 (good action): clip ratio to ≤ 1+ε
- If A < 0 (bad action): clip ratio to ≥ 1-ε

### Function Signature
\`\`\`python
def ppo_clip_objective(old_probs: np.ndarray, new_probs: np.ndarray,
                       advantages: np.ndarray, epsilon: float) -> float:
\`\`\`
    `,examples:[{input:"old_probs=[0.5], new_probs=[0.6], advantages=[1.0], epsilon=0.2",output:"1.2",explanation:"ratio = 1.2 is inside [0.8, 1.2], so min(1.2 * 1, clip(1.2) * 1) = 1.2"}],starterCode:`import numpy as np

def ppo_clip_objective(old_probs: np.ndarray, new_probs: np.ndarray,
                       advantages: np.ndarray, epsilon: float) -> float:
    """
    Compute PPO clipped surrogate objective.

    Args:
        old_probs: π_old(a|s) probabilities for taken actions
        new_probs: π_new(a|s) probabilities for taken actions
        advantages: Advantage estimates A_t
        epsilon: Clipping parameter (e.g., 0.2)

    Returns:
        PPO clipped objective (scalar, should be maximized)
    """
    old_probs = np.array(old_probs, dtype=float)
    new_probs = np.array(new_probs, dtype=float)
    advantages = np.array(advantages, dtype=float)

    # Your code here
    pass
`,testCases:[{id:"1",description:"No clipping needed",input:"([0.5, 0.5], [0.5, 0.5], [1.0, -1.0], 0.2)",expected:"0.0",hidden:!1},{id:"2",description:"Positive advantage, ratio > 1",input:"([0.5], [0.7], [1.0], 0.2)",expected:"1.2",hidden:!1},{id:"3",description:"Negative advantage, ratio > 1",input:"([0.5], [0.7], [-1.0], 0.2)",expected:"-1.4",hidden:!1},{id:"4",description:"Clipping in action",input:"([0.3], [0.6], [2.0], 0.2)",expected:"2.4",hidden:!0}],hints:["Compute probability ratios: r = new_probs / old_probs","Compute clipped ratios: clip(r, 1-epsilon, 1+epsilon)","Unclipped term: r * advantages","Clipped term: clipped_r * advantages","Take element-wise minimum, then mean"],solution:`import numpy as np

def ppo_clip_objective(old_probs: np.ndarray, new_probs: np.ndarray,
                       advantages: np.ndarray, epsilon: float) -> float:
    """
    Compute PPO clipped surrogate objective.
    """
    old_probs = np.array(old_probs, dtype=float)
    new_probs = np.array(new_probs, dtype=float)
    advantages = np.array(advantages, dtype=float)

    # Probability ratios
    ratios = new_probs / (old_probs + 1e-8)

    # Clipped ratios
    clipped_ratios = np.clip(ratios, 1 - epsilon, 1 + epsilon)

    # Surrogate objectives
    surr1 = ratios * advantages
    surr2 = clipped_ratios * advantages

    # PPO objective: min of clipped and unclipped
    objective = np.mean(np.minimum(surr1, surr2))

    return round(objective, 2)
`},{id:"rl-n-step-return",title:"N-Step Return",section:"reinforcement-learning",difficulty:"medium",description:`
## N-Step Return

Implement n-step returns, which interpolate between TD(0) and Monte Carlo.

### N-Step Return Formula
\`\`\`
G_t^{(n)} = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n * V(s_{t+n})
\`\`\`

### Trade-offs
| n | Method | Bias | Variance |
|---|--------|------|----------|
| 1 | TD(0) | High | Low |
| n | n-step | Medium | Medium |
| ∞ | MC | None | High |

### Function Signature
\`\`\`python
def n_step_return(rewards: np.ndarray, values: np.ndarray,
                  t: int, n: int, gamma: float) -> float:
\`\`\`

Compute n-step return starting from timestep t.
    `,examples:[{input:"rewards=[1,2,3,4], values=[0,0,0,0,10], t=0, n=2, gamma=0.9",output:"2.8",explanation:"G_0^(2) = 1 + 0.9*2 + 0.81*0 = 2.8"}],starterCode:`import numpy as np

def n_step_return(rewards: np.ndarray, values: np.ndarray,
                  t: int, n: int, gamma: float) -> float:
    """
    Compute n-step return from timestep t.

    Args:
        rewards: All rewards [r_0, r_1, ..., r_{T-1}]
        values: All values [V(s_0), ..., V(s_T)]
        t: Starting timestep
        n: Number of steps
        gamma: Discount factor

    Returns:
        N-step return G_t^(n)
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    T = len(rewards)

    # Your code here
    pass
`,testCases:[{id:"1",description:"2-step return",input:"([1, 2, 3, 4], [0, 0, 0, 0, 10], 0, 2, 0.9)",expected:"2.8",hidden:!1},{id:"2",description:"3-step return with bootstrap",input:"([1, 1, 1, 1], [0, 0, 0, 5, 0], 0, 3, 0.9)",expected:"6.36",hidden:!1},{id:"3",description:"1-step (TD) return",input:"([5, 2, 3], [1, 10, 5, 0], 0, 1, 0.9)",expected:"14.0",hidden:!0}],hints:["Sum discounted rewards from t to min(t+n-1, T-1)","Bootstrap with V(s_{t+n}) if t+n <= T","Handle boundary: if t+n > T, just sum remaining rewards","Discounts: γ^0, γ^1, ..., γ^{n-1} for rewards, γ^n for value"],solution:`import numpy as np

def n_step_return(rewards: np.ndarray, values: np.ndarray,
                  t: int, n: int, gamma: float) -> float:
    """
    Compute n-step return from timestep t.
    """
    rewards = np.array(rewards, dtype=float)
    values = np.array(values, dtype=float)
    T = len(rewards)

    G = 0.0

    # Sum discounted rewards
    for k in range(n):
        if t + k < T:
            G += (gamma ** k) * rewards[t + k]

    # Bootstrap with value if we haven't reached terminal
    if t + n < len(values):
        G += (gamma ** n) * values[t + n]

    return round(G, 2)
`}],gb=[{id:"rl-reward-modeling",title:"RLHF Reward Model Loss",section:"post-training",difficulty:"hard",description:`
## RLHF Reward Model Loss

Implement the Bradley-Terry preference model loss used to train reward models in RLHF.

### Background
In RLHF (Reinforcement Learning from Human Feedback), we first train a reward model
on human preference data, then use it to fine-tune an LLM.

### Bradley-Terry Model
Given a pair of responses (y_w, y_l) where y_w is preferred (winner):
\`\`\`
P(y_w > y_l) = σ(r(y_w) - r(y_l))
\`\`\`

Where σ is the sigmoid function and r(y) is the reward model output.

### Loss Function
\`\`\`
L = -E[log σ(r(y_w) - r(y_l))]
\`\`\`

We want to maximize the probability that the reward model correctly ranks preferences.

### Function Signature
\`\`\`python
def reward_model_loss(r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
\`\`\`

Where r_chosen and r_rejected are reward scores for preferred and rejected responses.
    `,examples:[{input:"r_chosen = [2.0, 1.5], r_rejected = [1.0, 0.5]",output:"0.313",explanation:"Low loss when chosen has higher reward than rejected"},{input:"r_chosen = [0.0], r_rejected = [0.0]",output:"0.693",explanation:"Equal rewards = 50% probability = -log(0.5) = 0.693"}],starterCode:`import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reward_model_loss(r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
    """
    Compute RLHF reward model loss (Bradley-Terry).

    Args:
        r_chosen: Reward scores for preferred/chosen responses
        r_rejected: Reward scores for rejected responses

    Returns:
        Average negative log-likelihood loss
    """
    r_chosen = np.array(r_chosen, dtype=float)
    r_rejected = np.array(r_rejected, dtype=float)

    # Your code here
    # Loss = -E[log σ(r_chosen - r_rejected)]
    pass
`,testCases:[{id:"1",description:"Clear preference",input:"([2.0, 1.5], [1.0, 0.5])",expected:"0.313",hidden:!1},{id:"2",description:"Equal scores",input:"([0.0], [0.0])",expected:"0.693",hidden:!1},{id:"3",description:"Large margin",input:"([5.0, 4.0, 3.0], [0.0, 0.0, 0.0])",expected:"0.024",hidden:!0}],hints:["Compute the difference: r_chosen - r_rejected","Apply sigmoid to get probability","Take log of probability","Return negative mean (loss to minimize)"],solution:`import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reward_model_loss(r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
    """
    Compute RLHF reward model loss (Bradley-Terry).
    """
    r_chosen = np.array(r_chosen, dtype=float)
    r_rejected = np.array(r_rejected, dtype=float)

    # Probability of correctly ranking: σ(r_chosen - r_rejected)
    logits = r_chosen - r_rejected
    probs = sigmoid(logits)

    # Negative log-likelihood
    loss = -np.mean(np.log(probs + 1e-10))

    return round(loss, 3)
`},{id:"rl-dpo-loss",title:"Direct Preference Optimization (DPO)",section:"post-training",difficulty:"hard",description:`
## Direct Preference Optimization (DPO) Loss

Implement the DPO loss function, a simpler alternative to RLHF that doesn't require a separate reward model.

### Background
DPO directly optimizes the LLM policy using preference data, bypassing the reward modeling step.

### DPO Loss
\`\`\`
L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                    - log π_θ(y_l|x) + log π_ref(y_l|x)))]
\`\`\`

Simplified:
\`\`\`
L_DPO = -E[log σ(β * (Δ_w - Δ_l))]
\`\`\`
Where Δ = log π_θ(y|x) - log π_ref(y|x) (log-ratio of policy to reference)

### Parameters
- β: Temperature parameter (typically 0.1 to 0.5)
- π_θ: Policy being trained
- π_ref: Reference policy (frozen)
- y_w: Preferred (winner) response
- y_l: Rejected (loser) response

### Function Signature
\`\`\`python
def dpo_loss(logp_chosen: np.ndarray, logp_rejected: np.ndarray,
             ref_logp_chosen: np.ndarray, ref_logp_rejected: np.ndarray,
             beta: float) -> float:
\`\`\`
    `,examples:[{input:"logp_chosen=[-1], logp_rejected=[-2], ref_logp_chosen=[-1.5], ref_logp_rejected=[-1.5], beta=0.1",output:"0.644",explanation:"Policy prefers chosen over reference"}],starterCode:`import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dpo_loss(logp_chosen: np.ndarray, logp_rejected: np.ndarray,
             ref_logp_chosen: np.ndarray, ref_logp_rejected: np.ndarray,
             beta: float) -> float:
    """
    Compute Direct Preference Optimization (DPO) loss.

    Args:
        logp_chosen: Log prob of chosen response under policy π_θ
        logp_rejected: Log prob of rejected response under policy π_θ
        ref_logp_chosen: Log prob of chosen under reference π_ref
        ref_logp_rejected: Log prob of rejected under reference π_ref
        beta: Temperature parameter

    Returns:
        DPO loss
    """
    logp_chosen = np.array(logp_chosen, dtype=float)
    logp_rejected = np.array(logp_rejected, dtype=float)
    ref_logp_chosen = np.array(ref_logp_chosen, dtype=float)
    ref_logp_rejected = np.array(ref_logp_rejected, dtype=float)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Policy prefers chosen",input:"([-1.0], [-2.0], [-1.5], [-1.5], 0.1)",expected:"0.644",hidden:!1},{id:"2",description:"Equal preferences",input:"([-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], 0.5)",expected:"0.693",hidden:!1},{id:"3",description:"Strong preference signal",input:"([0.0], [-5.0], [-2.0], [-3.0], 0.1)",expected:"0.513",hidden:!0}],hints:["Compute log-ratios: Δ_w = logp_chosen - ref_logp_chosen","Compute log-ratios: Δ_l = logp_rejected - ref_logp_rejected","DPO logits: β * (Δ_w - Δ_l)","Loss: -mean(log(sigmoid(logits)))"],solution:`import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dpo_loss(logp_chosen: np.ndarray, logp_rejected: np.ndarray,
             ref_logp_chosen: np.ndarray, ref_logp_rejected: np.ndarray,
             beta: float) -> float:
    """
    Compute Direct Preference Optimization (DPO) loss.
    """
    logp_chosen = np.array(logp_chosen, dtype=float)
    logp_rejected = np.array(logp_rejected, dtype=float)
    ref_logp_chosen = np.array(ref_logp_chosen, dtype=float)
    ref_logp_rejected = np.array(ref_logp_rejected, dtype=float)

    # Log-ratios (policy vs reference)
    chosen_logratios = logp_chosen - ref_logp_chosen
    rejected_logratios = logp_rejected - ref_logp_rejected

    # DPO logits
    logits = beta * (chosen_logratios - rejected_logratios)

    # Negative log-sigmoid loss
    loss = -np.mean(np.log(sigmoid(logits) + 1e-10))

    return round(loss, 3)
`},{id:"rl-kl-penalty",title:"KL Divergence Penalty (RLHF)",section:"post-training",difficulty:"medium",description:`
## KL Divergence Penalty for RLHF

Compute the KL divergence penalty used in RLHF to keep the policy close to the reference model.

### Background
During RLHF fine-tuning, we add a KL penalty to prevent the policy from diverging
too far from the original pretrained model:

\`\`\`
reward_total = reward - β * KL(π_θ || π_ref)
\`\`\`

### KL Divergence (per token)
\`\`\`
KL = E[log π_θ(a|s) - log π_ref(a|s)]
   = E[log(π_θ/π_ref)]
\`\`\`

For a sequence, we typically compute the average KL per token.

### Why KL Penalty?
- Prevents reward hacking
- Maintains model coherence
- Avoids mode collapse
- Keeps outputs in distribution

### Function Signature
\`\`\`python
def kl_penalty(logp_policy: np.ndarray, logp_reference: np.ndarray,
               beta: float) -> float:
\`\`\`

Returns: KL divergence multiplied by β
    `,examples:[{input:"logp_policy = [-1, -2], logp_reference = [-1.5, -2.5], beta = 0.1",output:"0.05",explanation:"KL = mean([0.5, 0.5]) = 0.5, penalty = 0.1 * 0.5 = 0.05"}],starterCode:`import numpy as np

def kl_penalty(logp_policy: np.ndarray, logp_reference: np.ndarray,
               beta: float) -> float:
    """
    Compute KL divergence penalty for RLHF.

    Args:
        logp_policy: Log probs from current policy π_θ
        logp_reference: Log probs from reference policy π_ref
        beta: KL penalty coefficient

    Returns:
        KL penalty = β * mean(KL divergence)
    """
    logp_policy = np.array(logp_policy, dtype=float)
    logp_reference = np.array(logp_reference, dtype=float)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Positive divergence",input:"([-1.0, -2.0], [-1.5, -2.5], 0.1)",expected:"0.05",hidden:!1},{id:"2",description:"No divergence",input:"([-1.0, -1.0], [-1.0, -1.0], 0.1)",expected:"0.0",hidden:!1},{id:"3",description:"Higher beta",input:"([-0.5, -1.0, -1.5], [-1.0, -1.5, -2.0], 0.5)",expected:"0.25",hidden:!0}],hints:["KL divergence per token: log(π_θ) - log(π_ref)","Take the mean over all tokens","Multiply by β to get the penalty","KL should be non-negative (though approximation may give small negatives)"],solution:`import numpy as np

def kl_penalty(logp_policy: np.ndarray, logp_reference: np.ndarray,
               beta: float) -> float:
    """
    Compute KL divergence penalty for RLHF.
    """
    logp_policy = np.array(logp_policy, dtype=float)
    logp_reference = np.array(logp_reference, dtype=float)

    # KL divergence: log(π_θ/π_ref) = log(π_θ) - log(π_ref)
    kl_div = logp_policy - logp_reference

    # Mean KL (per token)
    mean_kl = np.mean(kl_div)

    # KL penalty
    penalty = beta * mean_kl

    return round(penalty, 2)
`},{id:"rl-ppo-llm-objective",title:"PPO-RLHF Objective",section:"post-training",difficulty:"hard",description:`
## PPO Objective for LLM Fine-tuning (RLHF)

Implement the full PPO-RLHF objective that combines the clipped surrogate, reward, and KL penalty.

The KL penalty is applied through the reward (InstructGPT-style): \`advantage = reward - beta * KL\`, where \`KL = new_logp - ref_logp\`. The \`'objective'\` is the clipped surrogate computed with that advantage; \`'kl_penalty'\` reports \`beta * mean(KL)\` for monitoring.

### RLHF PPO Objective
\`\`\`
L = E[min(r_t * Â_t, clip(r_t, 1-ε, 1+ε) * Â_t) - β * KL]
\`\`\`

Where:
- r_t = π_θ(a|s) / π_old(a|s) is the probability ratio
- Â_t is the advantage (reward - baseline, typically from reward model)
- ε is the PPO clipping parameter (0.2)
- β is the KL penalty coefficient
- KL = log π_θ - log π_ref

### RLHF Advantage
In RLHF, the advantage is often simplified to:
\`\`\`
Â = R(x, y) - β * KL(π_θ || π_ref)
\`\`\`

### Function Signature
\`\`\`python
def ppo_rlhf_objective(old_logp: np.ndarray, new_logp: np.ndarray,
                       ref_logp: np.ndarray, rewards: np.ndarray,
                       epsilon: float, beta: float) -> dict:
\`\`\`

### Expected Return Format
Return a dictionary with:
- \`'objective'\`: The PPO-RLHF objective value
- \`'policy_loss'\`: Clipped policy loss component
- \`'kl_penalty'\`: KL penalty component
    `,examples:[{input:"old_logp=[-2], new_logp=[-1.8], ref_logp=[-2], rewards=[1.0], ε=0.2, β=0.1",output:"{'objective': ..., 'policy_loss': ..., 'kl_penalty': ...}",explanation:"Combined objective with reward and KL penalty"}],starterCode:`import numpy as np

def ppo_rlhf_objective(old_logp: np.ndarray, new_logp: np.ndarray,
                       ref_logp: np.ndarray, rewards: np.ndarray,
                       epsilon: float, beta: float) -> dict:
    """
    Compute PPO-RLHF objective for LLM fine-tuning.

    Args:
        old_logp: Log probs from old policy (before update)
        new_logp: Log probs from current policy
        ref_logp: Log probs from reference (pretrained) model
        rewards: Rewards from reward model
        epsilon: PPO clipping parameter
        beta: KL penalty coefficient

    Returns:
        Dictionary with 'objective', 'policy_loss', 'kl_penalty'
    """
    old_logp = np.array(old_logp, dtype=float)
    new_logp = np.array(new_logp, dtype=float)
    ref_logp = np.array(ref_logp, dtype=float)
    rewards = np.array(rewards, dtype=float)

    # Your code here
    pass
`,testCases:[{id:"1",description:"Positive reward, small update",input:"(lambda r: bool(abs(r['objective'] - 1.18) < 0.1 and abs(r['kl_penalty'] - 0.02) < 0.01))(ppo_rlhf_objective([-2.0], [-1.8], [-2.0], [1.0], 0.2, 0.1))",expected:"True",hidden:!1},{id:"2",description:"No policy change",input:"(lambda r: bool(abs(r['objective'] - 1.0) < 0.01 and abs(r['kl_penalty']) < 0.01))(ppo_rlhf_objective([-1.0], [-1.0], [-1.0], [1.0], 0.2, 0.1))",expected:"True",hidden:!1},{id:"3",description:"Clipping kicks in for a large ratio",input:'(lambda r: (round(float(r["objective"]), 3), round(float(r["kl_penalty"]), 3)))(ppo_rlhf_objective([-1.0, -2.0], [-0.5, -2.0], [-1.0, -2.0], [1.0, -1.0], 0.2, 0.1))',expected:"(0.07, 0.03)",hidden:!0}],hints:["Compute ratio: exp(new_logp - old_logp)","Compute KL: new_logp - ref_logp","Compute advantage: rewards - beta * KL","Apply PPO clipping to ratio * advantage","The KL penalty is already inside the advantage, so the objective is the clipped surrogate (report beta * mean KL separately)"],solution:`import numpy as np

def ppo_rlhf_objective(old_logp: np.ndarray, new_logp: np.ndarray,
                       ref_logp: np.ndarray, rewards: np.ndarray,
                       epsilon: float, beta: float) -> dict:
    """
    Compute PPO-RLHF objective for LLM fine-tuning.
    """
    old_logp = np.array(old_logp, dtype=float)
    new_logp = np.array(new_logp, dtype=float)
    ref_logp = np.array(ref_logp, dtype=float)
    rewards = np.array(rewards, dtype=float)

    # Probability ratio
    ratio = np.exp(new_logp - old_logp)

    # KL divergence from reference
    kl = new_logp - ref_logp

    # KL penalty
    kl_penalty = beta * np.mean(kl)

    # Advantage: reward minus KL penalty
    advantage = rewards - beta * kl

    # Clipped ratio
    clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)

    # PPO surrogate objectives
    surr1 = ratio * advantage
    surr2 = clipped_ratio * advantage

    # Policy loss (take minimum for conservative update)
    policy_loss = np.mean(np.minimum(surr1, surr2))

    # Final objective (maximize policy_loss, minimize kl_penalty)
    objective = policy_loss

    return {
        'objective': round(float(objective), 2),
        'policy_loss': round(float(policy_loss), 2),
        'kl_penalty': round(float(kl_penalty), 2),
    }
`}],_b=[{id:"e2e-mlp",title:"E2E: 2-Layer MLP with Backprop",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End 2-Layer MLP

Implement a complete 2-layer MLP with forward pass, backward pass, and training loop.

### Architecture
\`\`\`
Input (batch, input_dim)
  ↓
Linear(input_dim, hidden_dim) + ReLU
  ↓
Linear(hidden_dim, output_dim)
  ↓
Softmax + Cross-Entropy Loss
\`\`\`

### Forward Pass
\`\`\`python
h = relu(X @ W1 + b1)
logits = h @ W2 + b2
loss = cross_entropy(softmax(logits), y)
\`\`\`

### Backward Pass (Chain Rule)
\`\`\`python
# Output layer
d_logits = softmax(logits) - y_onehot  # (batch, output_dim)
d_W2 = h.T @ d_logits
d_b2 = d_logits.sum(axis=0)

# Hidden layer
d_h = d_logits @ W2.T
d_relu = d_h * (h > 0)  # ReLU derivative
d_W1 = X.T @ d_relu
d_b1 = d_relu.sum(axis=0)
\`\`\`

### Task
Implement forward, backward, and training step.
    `,examples:[{input:"MLP(784, 128, 10) on MNIST-like data",output:"Trained model with decreasing loss",explanation:"Full training loop"}],starterCode:`import numpy as np

class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize 2-layer MLP with random weights."""
        np.random.seed(42)
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

        # Cache for backward pass
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            X: Input (batch, input_dim)

        Returns:
            logits: Output (batch, output_dim)
        """
        # Your code here
        pass

    def compute_loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model output (batch, output_dim)
            y: True labels (batch,) as integers

        Returns:
            loss: Scalar loss value
        """
        # Your code here
        pass

    def backward(self, y: np.ndarray) -> dict:
        """
        Backward pass.

        Args:
            y: True labels (batch,) as integers

        Returns:
            gradients: Dictionary with dW1, db1, dW2, db2
        """
        # Your code here
        pass

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """
        Single training step: forward, loss, backward, update.

        Returns:
            loss: Loss value before update
        """
        # Your code here
        pass


def train_mlp(X_train, y_train, epochs=100, lr=0.01):
    """Train MLP and return loss history."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Forward pass shape",input:"MLP(784, 128, 10).forward(np.random.randn(4, 784)).shape",expected:"(4, 10)",hidden:!1},{id:"2",description:"Loss decreases",input:"(lambda: (m := MLP(10, 8, 3), X := np.random.randn(16, 10), y := np.array([0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0]), losses := [m.train_step(X, y) for _ in range(50)], losses[-1] < losses[0])[-1])()",expected:"True",hidden:!1},{id:"3",description:"Gradients computed",input:'(lambda: (m := MLP(10, 8, 3), X := np.random.randn(4, 10), y := np.array([0,1,2,0]), _ := m.forward(X), _ := m.compute_loss(m.cache["logits"], y), g := m.backward(y), all(k in g for k in ["dW1", "db1", "dW2", "db2"]))[-1])()',expected:"True",hidden:!0}],hints:["Cache X, h (hidden activations), and logits in forward pass","d_logits = probs - y_onehot (softmax gradient combined with cross-entropy)","ReLU gradient: 1 if x > 0, else 0","Update: W = W - lr * dW"],solution:`import numpy as np

class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Layer 1: Linear + ReLU
        z1 = X @ self.W1 + self.b1
        h = self.relu(z1)

        # Layer 2: Linear
        logits = h @ self.W2 + self.b2

        # Cache for backward
        self.cache = {'X': X, 'z1': z1, 'h': h, 'logits': logits}

        return logits

    def compute_loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        probs = self.softmax(logits)
        batch_size = logits.shape[0]
        correct_probs = probs[np.arange(batch_size), y]
        loss = -np.mean(np.log(np.clip(correct_probs, 1e-15, 1)))
        self.cache['probs'] = probs
        return loss

    def backward(self, y: np.ndarray) -> dict:
        X = self.cache['X']
        h = self.cache['h']
        z1 = self.cache['z1']
        probs = self.cache['probs']
        batch_size = X.shape[0]

        # One-hot encode targets
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(batch_size), y] = 1

        # Output layer gradients
        d_logits = (probs - y_onehot) / batch_size
        dW2 = h.T @ d_logits
        db2 = d_logits.sum(axis=0)

        # Hidden layer gradients
        d_h = d_logits @ self.W2.T
        d_relu = d_h * (z1 > 0)  # ReLU derivative
        dW1 = X.T @ d_relu
        db1 = d_relu.sum(axis=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        logits = self.forward(X)
        loss = self.compute_loss(logits, y)
        grads = self.backward(y)

        # Update weights
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']

        return loss


def train_mlp(X_train, y_train, epochs=100, lr=0.01):
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    model = MLP(input_dim, 64, output_dim)

    losses = []
    for _ in range(epochs):
        loss = model.train_step(X_train, y_train, lr)
        losses.append(loss)

    return model, losses
`},{id:"e2e-transformer",title:"E2E: Transformer Encoder",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End Transformer Encoder

Implement a complete Transformer encoder block.

### Architecture
\`\`\`
Input Embeddings + Positional Encoding
  ↓
Multi-Head Self-Attention
  ↓
Add & LayerNorm (Residual)
  ↓
Feed-Forward Network (FFN)
  ↓
Add & LayerNorm (Residual)
  ↓
Output
\`\`\`

### Components
1. **Positional Encoding**: Sinusoidal position embeddings
2. **Multi-Head Attention**: Parallel attention heads
3. **FFN**: Two linear layers with ReLU
4. **Layer Normalization**: Normalize across features
5. **Residual Connections**: Add input to output

### Task
Implement all components and combine into encoder block.
    `,examples:[{input:"TransformerEncoder(d_model=64, nhead=8, dim_ff=256)",output:"Encoded sequence (batch, seq_len, d_model)",explanation:"Complete encoder forward pass"}],starterCode:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization."""
    # Your code here
    pass


def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding."""
    # Your code here
    pass


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention."""
    # Your code here
    pass


def multi_head_attention(Q, K, V, num_heads):
    """Multi-head attention."""
    # Your code here
    pass


def feed_forward(x, W1, b1, W2, b2):
    """Position-wise feed-forward network: Linear -> ReLU -> Linear"""
    # Your code here
    pass


class TransformerEncoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """
        Initialize transformer encoder block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
        """
        np.random.seed(42)
        self.d_model = d_model
        self.num_heads = num_heads

        # Initialize weights
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through encoder block.

        Args:
            x: Input (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Your code here
        pass


def transformer_encoder(x, num_layers=2, d_model=64, num_heads=8, d_ff=256):
    """Stack multiple encoder blocks."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Output shape preserved",input:"transformer_encoder(np.random.randn(2, 10, 64), num_layers=2, d_model=64, num_heads=8, d_ff=256).shape",expected:"(2, 10, 64)",hidden:!1},{id:"2",description:"Attention weights sum to 1",input:"(lambda: (Q := np.random.randn(2, 5, 64), K := np.random.randn(2, 5, 64), V := np.random.randn(2, 5, 64), out_attn := scaled_dot_product_attention(Q, K, V), bool(np.allclose(out_attn[1].sum(axis=-1), 1.0)))[-1])()",expected:"True",hidden:!1},{id:"3",description:"Layer norm output is normalized per position",input:"(lambda x: (lambda out: bool(np.allclose(out.mean(-1), 0, atol=1e-5) and np.allclose(out.std(-1), 1, atol=1e-2)))(layer_norm(x, np.ones(16), np.zeros(16))))(np.random.randn(2, 5, 16) * 3 + 1)",expected:"True",hidden:!0}],hints:["Attention: softmax(QK^T / sqrt(d_k)) @ V","Layer norm: (x - mean) / sqrt(var + eps) * gamma + beta","Residual: output = LayerNorm(x + Sublayer(x))","FFN: max(0, xW1 + b1)W2 + b2"],solution:`import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    return PE


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attention = softmax(scores, axis=-1)
    return attention @ V, attention


def multi_head_attention(Q, K, V, num_heads):
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q = Q.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    d_k = Q.shape[-1]
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(d_k)
    attention = softmax(scores, axis=-1)
    output = np.einsum('bhqk,bhkd->bhqd', attention, V)

    output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    return output


def feed_forward(x, W1, b1, W2, b2):
    hidden = np.maximum(0, x @ W1 + b1)
    return hidden @ W2 + b2


class TransformerEncoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        np.random.seed(42)
        self.d_model = d_model
        self.num_heads = num_heads

        # Layer norm parameters
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

        # FFN parameters
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Multi-head self-attention + residual + layer norm
        attn_output = multi_head_attention(x, x, x, self.num_heads)
        x = layer_norm(x + attn_output, self.ln1_gamma, self.ln1_beta)

        # Feed-forward + residual + layer norm
        ff_output = feed_forward(x, self.W1, self.b1, self.W2, self.b2)
        x = layer_norm(x + ff_output, self.ln2_gamma, self.ln2_beta)

        return x


def transformer_encoder(x, num_layers=2, d_model=64, num_heads=8, d_ff=256):
    batch, seq_len, _ = x.shape

    # Add positional encoding
    PE = positional_encoding(seq_len, d_model)
    x = x + PE

    # Stack encoder blocks
    for _ in range(num_layers):
        block = TransformerEncoderBlock(d_model, num_heads, d_ff)
        x = block.forward(x)

    return x
`},{id:"e2e-vae",title:"E2E: Variational Autoencoder",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End Variational Autoencoder

Implement a complete VAE with encoder, decoder, reparameterization, and ELBO loss.

### Architecture
\`\`\`
Encoder: x → h → (μ, log_σ²)
         ↓
Reparameterization: z = μ + σ * ε, ε ~ N(0,1)
         ↓
Decoder: z → h → x_reconstructed
\`\`\`

### Loss (ELBO)
\`\`\`
L = Reconstruction Loss + KL Divergence
L = ||x - x_hat||² + KL(q(z|x) || p(z))
KL = -0.5 * sum(1 + log_var - μ² - exp(log_var))
\`\`\`

### Task
Implement encoder, decoder, reparameterization trick, and loss computation.
    `,examples:[{input:"VAE(input_dim=784, latent_dim=20)",output:"Reconstructed images + latent samples",explanation:"Full VAE forward pass"}],starterCode:`import numpy as np

class VAE:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Initialize VAE.

        Args:
            input_dim: Input dimension (e.g., 784 for MNIST)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
        """
        np.random.seed(42)

        # Encoder weights
        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_enc = np.zeros(hidden_dim)

        # Latent space (mean and log variance)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)

        # Decoder weights
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_dec2 = np.zeros(input_dim)

        self.latent_dim = latent_dim

    def encode(self, x: np.ndarray) -> tuple:
        """
        Encode input to latent distribution parameters.

        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        # Your code here
        pass

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        # Your code here
        pass

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent vector to reconstruction.
        """
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> dict:
        """
        Full forward pass.

        Returns:
            Dictionary with mu, log_var, z, x_recon
        """
        # Your code here
        pass

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     mu: np.ndarray, log_var: np.ndarray) -> dict:
        """
        Compute VAE loss (ELBO).

        Returns:
            Dictionary with total_loss, recon_loss, kl_loss
        """
        # Your code here
        pass

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Generate samples by decoding random latent vectors.
        """
        # Your code here
        pass
`,testCases:[{id:"1",description:"Reconstruction shape",input:'VAE(784, 256, 20).forward(np.random.randn(4, 784))["x_recon"].shape',expected:"(4, 784)",hidden:!1},{id:"2",description:"KL divergence non-negative",input:'(lambda: (vae := VAE(784, 256, 20), x := np.random.randn(4, 784), out := vae.forward(x), loss := vae.compute_loss(x, out["x_recon"], out["mu"], out["log_var"]), bool(loss["kl_loss"] >= 0))[-1])()',expected:"True",hidden:!1},{id:"3",description:"Reparameterize uses mu and std (eps=0 limit)",input:"(lambda v: bool(np.allclose(v.reparameterize(np.full((2, 3), 1.5), np.full((2, 3), -1e9)), 1.5)))(VAE(8, 4, 3))",expected:"True",hidden:!0}],hints:["Encoder: h = relu(x @ W + b), then mu = h @ W_mu, logvar = h @ W_logvar","Reparameterize: std = exp(0.5 * log_var), z = mu + std * epsilon","KL = -0.5 * mean(1 + log_var - mu² - exp(log_var))","Reconstruction loss: MSE or BCE"],solution:`import numpy as np

class VAE:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        np.random.seed(42)

        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_enc = np.zeros(hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_dec2 = np.zeros(input_dim)
        self.latent_dim = latent_dim

    def encode(self, x: np.ndarray) -> tuple:
        h = np.maximum(0, x @ self.W_enc + self.b_enc)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar
        return mu, log_var

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def decode(self, z: np.ndarray) -> np.ndarray:
        h = np.maximum(0, z @ self.W_dec1 + self.b_dec1)
        x_recon = 1 / (1 + np.exp(-(h @ self.W_dec2 + self.b_dec2)))  # Sigmoid
        return x_recon

    def forward(self, x: np.ndarray) -> dict:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return {'mu': mu, 'log_var': log_var, 'z': z, 'x_recon': x_recon}

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     mu: np.ndarray, log_var: np.ndarray) -> dict:
        # Reconstruction loss (MSE)
        recon_loss = np.mean((x - x_recon) ** 2)

        # KL divergence
        kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))

        total_loss = recon_loss + kl_loss

        return {'total_loss': total_loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}

    def sample(self, num_samples: int) -> np.ndarray:
        z = np.random.randn(num_samples, self.latent_dim)
        return self.decode(z)
`},{id:"e2e-vqvae",title:"E2E: Vector Quantized VAE",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End VQ-VAE

Implement a complete VQ-VAE (Vector Quantized Variational Autoencoder) with encoder, vector quantization, and decoder.

### Architecture
\`\`\`
Encoder: x → z_e (continuous latents)
           ↓
Vector Quantization: z_e → z_q (discrete latents from codebook)
           ↓
Decoder: z_q → x_reconstructed
\`\`\`

### Key Difference from VAE
- VAE: Continuous latent space with KL regularization
- VQ-VAE: Discrete latent space using learned codebook

### Vector Quantization
1. Encoder outputs continuous vectors z_e
2. Find nearest codebook entry for each spatial position
3. Replace z_e with quantized z_q from codebook
4. Use straight-through estimator for gradients

### Loss Function
\`\`\`
L = L_reconstruction + L_codebook + β * L_commitment

L_reconstruction = ||x - x_hat||²
L_codebook = ||sg[z_e] - e||²  (moves codebook toward encoder output)
L_commitment = ||z_e - sg[e]||²  (commits encoder to codebook)
\`\`\`
Where sg[] is stop-gradient.

### Codebook EMA Update (Alternative)
Instead of gradient updates, codebook can be updated with exponential moving average:
\`\`\`
N_i = γ * N_i + (1 - γ) * n_i     (count of assignments)
m_i = γ * m_i + (1 - γ) * sum(z_e assigned to i)
e_i = m_i / N_i
\`\`\`

### Task
Implement encoder, vector quantization layer, decoder, and all loss components.
    `,examples:[{input:"VQVAE(input_dim=784, num_embeddings=512, embedding_dim=64)",output:"Reconstructed images + discrete codes",explanation:"Full VQ-VAE forward pass"}],starterCode:`import numpy as np

class VectorQuantizer:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Vector Quantization layer.

        Args:
            num_embeddings: Size of codebook (K)
            embedding_dim: Dimension of each embedding
        """
        np.random.seed(42)
        # Initialize codebook with small random values
        self.embedding = np.random.randn(num_embeddings, embedding_dim) * 0.1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def quantize(self, z_e: np.ndarray) -> tuple:
        """
        Quantize encoder output to nearest codebook entries.

        Args:
            z_e: Encoder output (batch, height, width, embedding_dim)

        Returns:
            z_q: Quantized vectors (same shape as z_e)
            indices: Codebook indices (batch, height, width)
            distances: Distances to nearest embeddings
        """
        # Your code here
        pass

    def compute_loss(self, z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        """
        Compute VQ losses.

        Args:
            z_e: Encoder output
            z_q: Quantized vectors
            beta: Commitment loss weight

        Returns:
            Dictionary with codebook_loss and commitment_loss
        """
        # Your code here
        pass


class VQVAE:
    def __init__(self, input_channels: int, hidden_dim: int,
                 num_embeddings: int, embedding_dim: int):
        """
        Initialize VQ-VAE.

        Args:
            input_channels: Number of input channels
            hidden_dim: Hidden layer dimension
            num_embeddings: Codebook size
            embedding_dim: Embedding dimension
        """
        np.random.seed(42)

        # Encoder: conv layers to get spatial feature maps
        self.enc_conv1_w = np.random.randn(hidden_dim, input_channels, 4, 4) * 0.1
        self.enc_conv1_b = np.zeros(hidden_dim)
        self.enc_conv2_w = np.random.randn(embedding_dim, hidden_dim, 4, 4) * 0.1
        self.enc_conv2_b = np.zeros(embedding_dim)

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder: transposed conv to reconstruct
        self.dec_conv1_w = np.random.randn(hidden_dim, embedding_dim, 4, 4) * 0.1
        self.dec_conv1_b = np.zeros(hidden_dim)
        self.dec_conv2_w = np.random.randn(input_channels, hidden_dim, 4, 4) * 0.1
        self.dec_conv2_b = np.zeros(input_channels)

        self.embedding_dim = embedding_dim

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to continuous latent representation.

        Args:
            x: Input (batch, channels, height, width)

        Returns:
            z_e: Encoder output (batch, height', width', embedding_dim)
        """
        # Your code here (simplified: use strided conv or just reshape for demo)
        pass

    def decode(self, z_q: np.ndarray) -> np.ndarray:
        """
        Decode quantized latents to reconstruction.

        Args:
            z_q: Quantized vectors (batch, height, width, embedding_dim)

        Returns:
            x_recon: Reconstructed input
        """
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> dict:
        """
        Full forward pass.

        Returns:
            Dictionary with z_e, z_q, indices, x_recon
        """
        # Your code here
        pass

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        """
        Compute total VQ-VAE loss.

        Returns:
            Dictionary with total_loss, recon_loss, codebook_loss, commitment_loss
        """
        # Your code here
        pass

    def get_codebook_usage(self, indices: np.ndarray) -> np.ndarray:
        """
        Compute codebook usage statistics.

        Returns:
            usage: Count of each codebook entry used
        """
        # Your code here
        pass
`,testCases:[{id:"1",description:"Quantization shape",input:"(lambda: (vq := VectorQuantizer(512, 64), z_e := np.random.randn(2, 4, 4, 64), q := vq.quantize(z_e), q[0].shape == z_e.shape)[-1])()",expected:"True",hidden:!1},{id:"2",description:"Indices valid range",input:"(lambda: (vq := VectorQuantizer(512, 64), z_e := np.random.randn(2, 4, 4, 64), q := vq.quantize(z_e), bool(np.all(q[1] >= 0) and np.all(q[1] < 512)))[-1])()",expected:"True",hidden:!1},{id:"3",description:"Quantized from codebook",input:"(lambda: (vq := VectorQuantizer(512, 64), z_e := np.random.randn(2, 4, 4, 64), q := vq.quantize(z_e), bool(np.allclose(q[0][0, 0, 0], vq.embedding[q[1][0, 0, 0]])))[-1])()",expected:"True",hidden:!0},{id:"4",description:"Loss values correct",input:'(lambda: (vq := VectorQuantizer(512, 64), z_e := np.random.randn(2, 4, 4, 64), q := vq.quantize(z_e), losses := vq.compute_loss(z_e, q[0]), bool(losses["codebook_loss"] >= 0 and losses["commitment_loss"] >= 0))[-1])()',expected:"True",hidden:!0}],hints:["For quantization: flatten spatial dims, compute distances, find argmin","Distance: ||z - e||² = ||z||² + ||e||² - 2*z·e","Straight-through: z_q = z_e + stop_grad(z_q - z_e)","Codebook loss uses z_e detached, commitment uses z_q detached","For simplified encoder/decoder, use reshape operations"],solution:`import numpy as np

class VectorQuantizer:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        np.random.seed(42)
        self.embedding = np.random.randn(num_embeddings, embedding_dim) * 0.1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def quantize(self, z_e: np.ndarray) -> tuple:
        # z_e: (batch, H, W, D)
        batch, H, W, D = z_e.shape

        # Flatten spatial dimensions
        z_flat = z_e.reshape(-1, D)  # (N, D) where N = batch * H * W

        # Compute distances: ||z - e||² = ||z||² + ||e||² - 2*z·e
        z_sq = np.sum(z_flat ** 2, axis=1, keepdims=True)  # (N, 1)
        e_sq = np.sum(self.embedding ** 2, axis=1)          # (K,)
        cross = z_flat @ self.embedding.T                    # (N, K)
        distances = z_sq + e_sq - 2 * cross                  # (N, K)

        # Find nearest embedding
        indices_flat = np.argmin(distances, axis=1)          # (N,)

        # Get quantized vectors
        z_q_flat = self.embedding[indices_flat]              # (N, D)

        # Reshape back
        z_q = z_q_flat.reshape(batch, H, W, D)
        indices = indices_flat.reshape(batch, H, W)
        min_distances = np.min(distances, axis=1).reshape(batch, H, W)

        return z_q, indices, min_distances

    def compute_loss(self, z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        # Codebook loss: ||sg[z_e] - z_q||²
        # In practice, this updates the codebook to move toward encoder output
        codebook_loss = np.mean((z_e - z_q) ** 2)

        # Commitment loss: ||z_e - sg[z_q]||²
        # This commits the encoder output to stay close to codebook
        commitment_loss = beta * np.mean((z_e - z_q) ** 2)

        return {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }


class VQVAE:
    def __init__(self, input_channels: int, hidden_dim: int,
                 num_embeddings: int, embedding_dim: int):
        np.random.seed(42)

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Simplified: use linear projection instead of conv for demo
        # Encoder projects input to embedding space
        self.enc_w1 = np.random.randn(input_channels * 16, hidden_dim) * 0.1
        self.enc_b1 = np.zeros(hidden_dim)
        self.enc_w2 = np.random.randn(hidden_dim, embedding_dim) * 0.1
        self.enc_b2 = np.zeros(embedding_dim)

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder projects back
        self.dec_w1 = np.random.randn(embedding_dim, hidden_dim) * 0.1
        self.dec_b1 = np.zeros(hidden_dim)
        self.dec_w2 = np.random.randn(hidden_dim, input_channels * 16) * 0.1
        self.dec_b2 = np.zeros(input_channels * 16)

    def encode(self, x: np.ndarray) -> np.ndarray:
        # x: (batch, channels, H, W) - assume H=W=4 for simplicity
        batch = x.shape[0]

        # Flatten spatial dimensions
        x_flat = x.reshape(batch, -1)  # (batch, channels * H * W)

        # Encode
        h = np.maximum(0, x_flat @ self.enc_w1 + self.enc_b1)  # ReLU
        z_e = h @ self.enc_w2 + self.enc_b2  # (batch, embedding_dim)

        # Reshape to spatial format (batch, 2, 2, embedding_dim)
        z_e = z_e.reshape(batch, 1, 1, self.embedding_dim)

        return z_e

    def decode(self, z_q: np.ndarray) -> np.ndarray:
        batch = z_q.shape[0]

        # Flatten
        z_flat = z_q.reshape(batch, -1)  # (batch, embedding_dim)

        # Decode
        h = np.maximum(0, z_flat @ self.dec_w1 + self.dec_b1)  # ReLU
        x_recon = h @ self.dec_w2 + self.dec_b2

        # Reshape to image format
        x_recon = x_recon.reshape(batch, self.input_channels, 4, 4)

        return x_recon

    def forward(self, x: np.ndarray) -> dict:
        # Encode
        z_e = self.encode(x)

        # Quantize
        z_q, indices, distances = self.vq.quantize(z_e)

        # Straight-through estimator: gradient flows through z_q to z_e
        # z_q_st = z_e + (z_q - z_e).detach()
        # For forward pass, we just use z_q

        # Decode
        x_recon = self.decode(z_q)

        return {
            'z_e': z_e,
            'z_q': z_q,
            'indices': indices,
            'x_recon': x_recon
        }

    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray,
                     z_e: np.ndarray, z_q: np.ndarray, beta: float = 0.25) -> dict:
        # Reconstruction loss
        recon_loss = np.mean((x - x_recon) ** 2)

        # VQ losses
        vq_losses = self.vq.compute_loss(z_e, z_q, beta)

        # Total loss
        total_loss = recon_loss + vq_losses['codebook_loss'] + vq_losses['commitment_loss']

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'codebook_loss': vq_losses['codebook_loss'],
            'commitment_loss': vq_losses['commitment_loss']
        }

    def get_codebook_usage(self, indices: np.ndarray) -> np.ndarray:
        usage = np.bincount(indices.flatten(), minlength=self.vq.num_embeddings)
        return usage
`},{id:"e2e-diffusion",title:"E2E: Diffusion Model",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End Diffusion Model

Implement core components of a diffusion model (DDPM-style).

### Forward Process (Adding Noise)
\`\`\`
x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
\`\`\`

### Noise Schedule
\`\`\`
β_t: Linear from β_start to β_end
α_t = 1 - β_t
α_bar_t = cumulative product of α_t
\`\`\`

### Training Objective
\`\`\`
L = ||ε - ε_θ(x_t, t)||²
\`\`\`

Model learns to predict the noise added at timestep t.

### Reverse Process (Sampling)
\`\`\`
x_{t-1} = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-α_bar_t)) * ε_θ(x_t, t)) + σ_t * z
\`\`\`

### Task
Implement noise schedule, forward process, and reverse sampling.
    `,examples:[{input:"DiffusionModel(T=1000)",output:"Denoised samples",explanation:"Forward and reverse diffusion"}],starterCode:`import numpy as np

class DiffusionModel:
    def __init__(self, T: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        """
        Initialize diffusion model.

        Args:
            T: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        self.T = T

        # Compute noise schedule
        # Your code here
        pass

    def get_noise_schedule(self) -> dict:
        """Return computed noise schedule parameters."""
        # Your code here
        pass

    def forward_diffusion(self, x_0: np.ndarray, t: int) -> tuple:
        """
        Add noise to data at timestep t.

        Args:
            x_0: Original data (batch, ...)
            t: Timestep

        Returns:
            x_t: Noisy data
            noise: The noise that was added
        """
        np.random.seed(42)
        # Your code here
        pass

    def reverse_step(self, x_t: np.ndarray, t: int, predicted_noise: np.ndarray) -> np.ndarray:
        """
        Single reverse diffusion step.

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            predicted_noise: Noise predicted by model

        Returns:
            x_{t-1}: Less noisy data
        """
        # Your code here
        pass

    def compute_loss(self, true_noise: np.ndarray, predicted_noise: np.ndarray) -> float:
        """Compute MSE loss between true and predicted noise."""
        # Your code here
        pass

    def sample(self, shape: tuple, noise_predictor) -> np.ndarray:
        """
        Generate samples using reverse diffusion.

        Args:
            shape: Shape of samples to generate
            noise_predictor: Function that predicts noise given (x_t, t)

        Returns:
            Generated samples
        """
        # Your code here
        pass
`,testCases:[{id:"1",description:"Alpha bar decreases",input:'(lambda: (dm := DiffusionModel(1000), sched := dm.get_noise_schedule(), bool(sched["alpha_bars"][-1] < sched["alpha_bars"][0]))[-1])()',expected:"True",hidden:!1},{id:"2",description:"Forward noise shape",input:"(lambda: (dm := DiffusionModel(1000), x_0 := np.random.randn(4, 32), fd := dm.forward_diffusion(x_0, 500), fd[0].shape == x_0.shape)[-1])()",expected:"True",hidden:!1},{id:"3",description:"Forward diffusion matches the closed form",input:'(lambda dm, x0: (lambda r: bool(np.allclose(r[0], np.sqrt(dm.get_noise_schedule()["alpha_bars"][10]) * x0 + np.sqrt(1 - dm.get_noise_schedule()["alpha_bars"][10]) * r[1])))(dm.forward_diffusion(x0, 10)))(DiffusionModel(100), np.random.randn(3, 4))',expected:"True",hidden:!0}],hints:["betas = np.linspace(beta_start, beta_end, T)","alpha_bars = np.cumprod(1 - betas)","x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise","Reverse: use mean and add small noise (except at t=0)"],solution:`import numpy as np

class DiffusionModel:
    def __init__(self, T: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.T = T

        # Linear noise schedule
        self.betas = np.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

        # For reverse process
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1 - self.alpha_bars)
        self.sqrt_alphas = np.sqrt(self.alphas)

    def get_noise_schedule(self) -> dict:
        return {
            'betas': self.betas,
            'alphas': self.alphas,
            'alpha_bars': self.alpha_bars
        }

    def forward_diffusion(self, x_0: np.ndarray, t: int) -> tuple:
        np.random.seed(42)
        noise = np.random.randn(*x_0.shape)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

    def reverse_step(self, x_t: np.ndarray, t: int, predicted_noise: np.ndarray) -> np.ndarray:
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]

        # Mean of reverse distribution
        coef1 = 1 / np.sqrt(alpha)
        coef2 = beta / np.sqrt(1 - alpha_bar)
        mean = coef1 * (x_t - coef2 * predicted_noise)

        if t > 0:
            # Add noise (except at final step)
            sigma = np.sqrt(beta)
            noise = np.random.randn(*x_t.shape)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean

        return x_prev

    def compute_loss(self, true_noise: np.ndarray, predicted_noise: np.ndarray) -> float:
        return np.mean((true_noise - predicted_noise) ** 2)

    def sample(self, shape: tuple, noise_predictor) -> np.ndarray:
        # Start from pure noise
        x = np.random.randn(*shape)

        # Reverse diffusion
        for t in reversed(range(self.T)):
            predicted_noise = noise_predictor(x, t)
            x = self.reverse_step(x, t, predicted_noise)

        return x
`},{id:"e2e-cnn",title:"E2E: Convolutional Neural Network",section:"e2e-implementations",difficulty:"hard",description:`
## End-to-End CNN

Implement a complete CNN for image classification.

### Architecture
\`\`\`
Input (batch, channels, height, width)
  ↓
Conv2D(in_channels, 32, kernel=3) + ReLU
  ↓
MaxPool2D(2x2)
  ↓
Conv2D(32, 64, kernel=3) + ReLU
  ↓
MaxPool2D(2x2)
  ↓
Flatten
  ↓
Linear(flattened_size, 128) + ReLU
  ↓
Linear(128, num_classes)
  ↓
Softmax
\`\`\`

### Components
- 2D Convolution (no padding for simplicity)
- Max Pooling
- Flatten
- Fully Connected layers

### Task
Implement conv2d, maxpool, and combine into CNN.
    `,examples:[{input:"CNN on 28x28 grayscale images",output:"Class probabilities",explanation:"Full CNN forward pass"}],starterCode:`import numpy as np

def conv2d(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    2D convolution (no padding, stride=1).

    Args:
        x: Input (batch, in_channels, height, width)
        kernel: Weights (out_channels, in_channels, kH, kW)
        bias: Bias (out_channels,)

    Returns:
        output: (batch, out_channels, out_height, out_width)
    """
    # Your code here
    pass


def maxpool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """
    2D max pooling.

    Args:
        x: Input (batch, channels, height, width)
        pool_size: Size of pooling window

    Returns:
        output: Pooled tensor
    """
    # Your code here
    pass


def flatten(x: np.ndarray) -> np.ndarray:
    """Flatten all dimensions except batch."""
    # Your code here
    pass


class CNN:
    def __init__(self, input_shape: tuple, num_classes: int):
        """
        Initialize CNN.

        Args:
            input_shape: (channels, height, width)
            num_classes: Number of output classes
        """
        np.random.seed(42)
        in_channels, H, W = input_shape

        # Conv layer 1: in_channels -> 32
        self.conv1_w = np.random.randn(32, in_channels, 3, 3) * 0.1
        self.conv1_b = np.zeros(32)

        # Conv layer 2: 32 -> 64
        self.conv2_w = np.random.randn(64, 32, 3, 3) * 0.1
        self.conv2_b = np.zeros(64)

        # Calculate flattened size after convolutions and pooling
        # Your code to compute this
        pass

        # Fully connected layers
        # Your code here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through CNN.

        Args:
            x: Input images (batch, channels, height, width)

        Returns:
            logits: (batch, num_classes)
        """
        # Your code here
        pass


def test_cnn():
    """Test CNN with sample data."""
    # Your code here
    pass
`,testCases:[{id:"1",description:"Conv output shape",input:"conv2d(np.random.randn(2, 1, 8, 8), np.random.randn(32, 1, 3, 3), np.zeros(32)).shape",expected:"(2, 32, 6, 6)",hidden:!1},{id:"2",description:"Pool output shape",input:"maxpool2d(np.random.randn(2, 32, 8, 8), pool_size=2).shape",expected:"(2, 32, 4, 4)",hidden:!1},{id:"3",description:"conv2d values match a direct computation",input:"(lambda x, k: bool(np.isclose(conv2d(x, k, np.array([0.5]))[0, 0, 1, 2], np.sum(x[0, :, 1:4, 2:5] * k[0]) + 0.5)))(np.random.randn(1, 2, 6, 6), np.random.randn(1, 2, 3, 3))",expected:"True",hidden:!0}],hints:["Conv output size: (input - kernel + 1) for stride=1, no padding","Pool output size: input // pool_size","Use nested loops for convolution (or np.lib.stride_tricks)","Flatten: x.reshape(batch_size, -1)"],solution:`import numpy as np

def conv2d(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, in_ch, H, W = x.shape
    out_ch, _, kH, kW = kernel.shape
    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((batch, out_ch, out_H, out_W))

    for b in range(batch):
        for oc in range(out_ch):
            for i in range(out_H):
                for j in range(out_W):
                    receptive_field = x[b, :, i:i+kH, j:j+kW]
                    output[b, oc, i, j] = np.sum(receptive_field * kernel[oc]) + bias[oc]

    return output


def maxpool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
    batch, channels, H, W = x.shape
    out_H = H // pool_size
    out_W = W // pool_size

    output = np.zeros((batch, channels, out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * pool_size
            w_start = j * pool_size
            output[:, :, i, j] = np.max(
                x[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size],
                axis=(2, 3)
            )

    return output


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


class CNN:
    def __init__(self, input_shape: tuple, num_classes: int):
        np.random.seed(42)
        in_channels, H, W = input_shape

        # Conv layers
        self.conv1_w = np.random.randn(32, in_channels, 3, 3) * 0.1
        self.conv1_b = np.zeros(32)
        self.conv2_w = np.random.randn(64, 32, 3, 3) * 0.1
        self.conv2_b = np.zeros(64)

        # Calculate size after convolutions and pooling
        # After conv1 (3x3): H-2, W-2
        # After pool1 (2x2): (H-2)//2, (W-2)//2
        # After conv2 (3x3): (H-2)//2 - 2, (W-2)//2 - 2
        # After pool2 (2x2): ((H-2)//2 - 2)//2, ((W-2)//2 - 2)//2
        h1 = (H - 2) // 2
        w1 = (W - 2) // 2
        h2 = (h1 - 2) // 2
        w2 = (w1 - 2) // 2
        flat_size = 64 * h2 * w2

        # FC layers
        self.fc1_w = np.random.randn(flat_size, 128) * 0.1
        self.fc1_b = np.zeros(128)
        self.fc2_w = np.random.randn(128, num_classes) * 0.1
        self.fc2_b = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Conv block 1
        x = conv2d(x, self.conv1_w, self.conv1_b)
        x = np.maximum(0, x)  # ReLU
        x = maxpool2d(x)

        # Conv block 2
        x = conv2d(x, self.conv2_w, self.conv2_b)
        x = np.maximum(0, x)  # ReLU
        x = maxpool2d(x)

        # Flatten
        x = flatten(x)

        # FC layers
        x = np.maximum(0, x @ self.fc1_w + self.fc1_b)  # ReLU
        x = x @ self.fc2_w + self.fc2_b  # Logits

        return x


def test_cnn():
    # Test with MNIST-like data
    cnn = CNN(input_shape=(1, 28, 28), num_classes=10)
    x = np.random.randn(4, 1, 28, 28)
    logits = cnn.forward(x)
    return logits.shape == (4, 10)
`}],yb=[...eb,...tb,...nb,...rb,...ib,...ob,...ab,...sb,...lb,...ub,...db,...cb,...pb,...hb,...mb,...fb,...gb,..._b],os=yb.map(e=>hp[e.id]?{...e,pytorch:hp[e.id]}:e);function xb(e){return os.find(t=>t.id===e)}function bb(e){return os.filter(t=>t.section===e)}const Us="ml-interview-progress";function vb(e){var r;const t=new Map(os.map(i=>[i.id,i.section])),n={};for(const[i,o]of Object.entries(e??{}))if(!(!o||typeof o!="object"))for(const[a,s]of Object.entries(o)){const l=t.get(a)??i;n[l]={...n[l],[a]:{...(r=n[l])==null?void 0:r[a],...s}}}return n}const a0=C.createContext(null);function kb({children:e}){const[t,n]=C.useState(()=>{const c=localStorage.getItem(Us);return c?vb(JSON.parse(c)):{}});C.useEffect(()=>{localStorage.setItem(Us,JSON.stringify(t))},[t]);const r=C.useCallback((c,d,h,m)=>{n(f=>{var g;const x=(g=f[c])==null?void 0:g[d],S=(x==null?void 0:x.solvedIn)??[];return{...f,[c]:{...f[c],[d]:{...x,status:h,...h==="completed"&&m&&!S.includes(m)?{solvedIn:[...S,m]}:{},lastAttempt:new Date().toISOString()}}}})},[]),i=C.useCallback((c,d,h,m="numpy")=>{n(f=>{var S;const x=(S=f[c])==null?void 0:S[d];return{...f,[c]:{...f[c],[d]:{...x,status:(x==null?void 0:x.status)||"in_progress",...m==="numpy"?{code:h}:{codes:{...x==null?void 0:x.codes,[m]:h}},lastAttempt:new Date().toISOString()}}}})},[]),o=C.useCallback((c,d)=>{n(h=>{var f;const m=(f=h[c])==null?void 0:f[d];return m!=null&&m.solutionViewed?h:{...h,[c]:{...h[c],[d]:{...m,status:(m==null?void 0:m.status)||"in_progress",solutionViewed:!0}}}})},[]),a=C.useCallback((c,d)=>{var m;const h=(m=t[c])==null?void 0:m[d];return{status:(h==null?void 0:h.status)||"not_started",code:h==null?void 0:h.code,codes:(h==null?void 0:h.codes)??{},solvedIn:(h==null?void 0:h.solvedIn)??[],solutionViewed:(h==null?void 0:h.solutionViewed)??!1}},[t]),s=C.useCallback((c,d)=>{const h=t[c];if(!h||d===0)return 0;const m=Object.values(h).filter(f=>f.status==="completed").length;return Math.round(m/d*100)},[t]),l=C.useCallback(c=>{const d=c.reduce((m,f)=>m+f.problemCount,0);if(d===0)return 0;let h=0;for(const m of c){const f=t[m.id];f&&(h+=Object.values(f).filter(x=>x.status==="completed").length)}return Math.round(h/d*100)},[t]),u=C.useCallback(()=>{n({}),localStorage.removeItem(Us)},[]);return p.jsx(a0.Provider,{value:{progress:t,updateProblemStatus:r,saveProblemCode:i,markSolutionViewed:o,getProblemProgress:a,getSectionProgress:s,getOverallProgress:l,resetProgress:u},children:e})}function lo(){const e=C.useContext(a0);if(!e)throw new Error("useProgress must be used within a ProgressProvider");return e}function Le(e,t){const n=os.filter(r=>r.section===e).map(r=>r.id);return[...t.filter(r=>n.includes(r)),...n.filter(r=>!t.includes(r))]}const wb=[{id:"numpy-fundamentals",title:"NumPy & Tensor Fundamentals",description:"Arrays, indexing, broadcasting and vectorized math: the language every ML model is written in.",icon:"🔢",introduction:"\n# NumPy Fundamentals\n\nNumPy is the backbone of scientific computing in Python. Every ML library builds on it.\n\n## Key Concepts\n\n### Array Creation\n- `np.array()`: Create from lists\n- `np.zeros()`, `np.ones()`: Initialize with values\n- `np.arange()`, `np.linspace()`: Numeric ranges\n- `np.eye()`: Identity matrix\n\n### Indexing & Slicing\n- Basic: `arr[0]`, `arr[1:5]`\n- Advanced: `arr[[0, 2, 4]]`, `arr[arr > 0]`\n- Multi-dimensional: `arr[1, :]`, `arr[:, 2:4]`\n\n### Broadcasting\nArrays of different shapes can be combined:\n- (3, 4) + (4,) → broadcasts to (3, 4)\n- (3, 1) * (1, 4) → broadcasts to (3, 4)\n\n### Aggregations\n- `sum()`, `mean()`, `std()` - with axis parameter\n- `argmax()`, `argmin()` - find indices\n\n### Shape Manipulation\n- `reshape()`: Change dimensions\n- `transpose()`, `.T`: Swap axes\n- `flatten()`, `ravel()`: To 1D\n\nMaster these operations to write efficient ML code!\n\n---\n\n## Python Basics for Machine Learning\n\nNumPy is the foundation of nearly all machine learning in Python. Before diving into complex algorithms, you need to master array operations.\n\n### Key Concepts\n\n#### NumPy Arrays\n- **Creation**: `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`\n- **Shape manipulation**: `reshape()`, `flatten()`, `transpose()`\n- **Indexing**: Slicing, boolean indexing, fancy indexing\n\n#### Essential Operations\n- **Element-wise**: Addition, multiplication, mathematical functions\n- **Aggregations**: `sum()`, `mean()`, `std()`, `min()`, `max()`\n- **Broadcasting**: Automatic expansion of arrays for operations\n\n#### Why It Matters\nEvery ML algorithm processes data as arrays. Understanding these operations lets you:\n- Prepare data efficiently\n- Debug model issues\n- Optimize performance\n\nLet's practice with hands-on problems!\n    ",problems:Le("numpy-fundamentals",["numpy-array-creation","numpy-indexing","numpy-broadcasting","numpy-aggregations","numpy-reshape-transpose","numpy-array-sum","numpy-matrix-multiply","numpy-broadcast-add"])},{id:"einsum",title:"Einstein Summation (Einsum)",description:"Master einsum for elegant tensor operations.",icon:"∑",introduction:`
# Einstein Summation (Einsum)

Einsum is a powerful notation for expressing tensor operations concisely.

## Why Einsum?
- Express complex operations in one line
- Often faster than chained operations
- Essential for attention mechanisms

## Syntax
\`\`\`python
np.einsum('subscripts', operands)
\`\`\`

## Common Patterns

### Basic Operations
\`\`\`
'ij->'      Sum all elements
'ij->i'     Sum along axis 1 (row sums)
'ij->j'     Sum along axis 0 (column sums)
'ij->ji'    Transpose
'ii->i'     Extract diagonal
'ii->'      Trace (sum of diagonal)
\`\`\`

### Matrix Operations
\`\`\`
'ik,kj->ij'   Matrix multiplication (A @ B)
'ij,ij->ij'   Element-wise product (A * B)
'ij,ij->'     Frobenius inner product
'i,j->ij'     Outer product
'i,i->'       Dot product
\`\`\`

### Batch Operations
\`\`\`
'bij,bjk->bik'   Batch matrix multiply
'bqd,bkd->bqk'   Attention scores (Q @ K.T)
'bhqk,bhkd->bhqd' Multi-head attention output
\`\`\`

## Key Insight
Letters that appear on both sides are kept.
Letters that disappear are summed over (contracted).

Master einsum to write clean, efficient deep learning code!
    `,problems:Le("einsum",["einsum-basics","einsum-matrix-ops","einsum-batch-ops","einsum-advanced","einsum-vs-matmul"])},{id:"pytorch-basics",title:"PyTorch Essentials",description:"Tensors, autograd, nn.Module and the training loop, written with the real PyTorch API.",icon:"🔥",defaultLanguage:"pytorch",introduction:`
# PyTorch Basics

Learn PyTorch patterns and concepts. Since Pyodide doesn't support PyTorch, we implement these concepts in NumPy.

## Tensor Operations

### Creation
\`\`\`python
# PyTorch           # NumPy equivalent
torch.tensor()      np.array()
torch.zeros()       np.zeros()
torch.randn()       np.random.randn()
\`\`\`

### Operations
\`\`\`python
# PyTorch           # NumPy equivalent
x + y               x + y
x @ y               x @ y
x.sum(dim=1)        x.sum(axis=1)
x.view(2, -1)       x.reshape(2, -1)
\`\`\`

## Autograd Concepts

PyTorch tracks operations for automatic differentiation:
\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # dy/dx = 2x = 4
\`\`\`

We implement this manually to understand the math.

## Module Pattern
\`\`\`python
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        self.weight = Parameter(...)
        self.bias = Parameter(...)

    def forward(self, x):
        return x @ self.weight + self.bias
\`\`\`

## Loss Functions
- CrossEntropyLoss: Classification
- MSELoss: Regression
- BCELoss: Binary classification

Learn these patterns to understand PyTorch!
    `,problems:Le("pytorch-basics",["tensor-creation","tensor-operations","autograd-concepts","nn-modules","loss-functions"])},{id:"data-preprocessing",title:"Data Preprocessing",description:"Learn essential data cleaning and transformation techniques.",icon:"🧹",introduction:`
# Data Preprocessing

Real-world data is messy. Before training any model, you need to clean and transform your data.

## Key Concepts

### Handling Missing Data
- **Detection**: \`np.isnan()\`, \`pd.isnull()\`
- **Strategies**: Remove, fill with mean/median/mode, interpolate

### Feature Scaling
- **Normalization**: Scale to [0, 1] range using min-max scaling
- **Standardization**: Transform to zero mean, unit variance (z-score)
- **When to use**: Most algorithms need scaled features

### Encoding Categorical Data
- **One-Hot Encoding**: Convert categories to binary vectors
- **Label Encoding**: Map categories to integers
- **When to use**: Tree-based models handle label encoding; others need one-hot

### Why It Matters
Garbage in, garbage out. Proper preprocessing can:
- Improve model accuracy by 10-30%
- Prevent training failures
- Reduce overfitting

Let's practice these essential skills!
    `,problems:Le("data-preprocessing",["normalize-features","handle-missing-data","one-hot-encode"])},{id:"supervised-learning",title:"Supervised Learning",description:"Implement core supervised learning algorithms from scratch.",icon:"📊",introduction:`
# Supervised Learning

Supervised learning is when you train a model on labeled data to make predictions.

## Key Concepts

### Linear Regression
- **Goal**: Predict continuous values
- **Formula**: y = wx + b
- **Training**: Minimize mean squared error (MSE)
- **Gradient descent**: Update weights iteratively

### Logistic Regression
- **Goal**: Binary classification
- **Sigmoid function**: Maps outputs to [0, 1] probabilities
- **Loss**: Binary cross-entropy
- **Decision boundary**: Linear separation of classes

### Decision Trees
- **Concept**: Recursive binary splits based on features
- **Splitting criteria**: Gini impurity, Information gain
- **Advantages**: Interpretable, handles non-linear relationships

### Support Vector Machines (SVMs)
- **Goal**: Find the maximum-margin hyperplane
- **Hinge loss**: max(0, 1 - y * score) — penalizes misclassified and low-margin samples
- **Regularization**: L2 penalty on weights to maximize margin
- **Key insight**: Only "support vectors" (points near the boundary) affect the decision boundary

### The Interview Perspective
Interviewers often ask you to:
- Implement gradient descent from scratch
- Explain the math behind these algorithms
- Compute and interpret hinge loss
- Discuss trade-offs between algorithms

Let's build these algorithms!
    `,problems:Le("supervised-learning",["logistic-regression","binary-cross-entropy","hinge-loss","decision-tree-split","linear-regression-gd","logistic-regression-full","linear-svm"])},{id:"unsupervised-learning",title:"Unsupervised Learning",description:"Implement clustering and dimensionality reduction algorithms.",icon:"🔍",introduction:`
# Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.

## Key Concepts

### K-Means Clustering
- **Goal**: Group data into K clusters
- **Algorithm**:
  1. Initialize K centroids randomly
  2. Assign points to nearest centroid
  3. Update centroids as cluster means
  4. Repeat until convergence

### Principal Component Analysis (PCA)
- **Goal**: Reduce dimensionality while preserving variance
- **Steps**:
  1. Center the data (subtract mean)
  2. Compute covariance matrix
  3. Find eigenvectors/eigenvalues
  4. Project onto top components

### Why These Matter
- **K-Means**: Customer segmentation, image compression
- **PCA**: Feature reduction, visualization, noise removal

### Interview Tips
- Be ready to implement K-means from scratch
- Explain variance explained by components
- Discuss limitations (K-means assumes spherical clusters)

Let's implement these algorithms!
    `,problems:Le("unsupervised-learning",["kmeans-clustering","pca-implementation"])},{id:"model-evaluation",title:"Model Evaluation",description:"Learn metrics and techniques to evaluate ML models properly.",icon:"📈",introduction:`
# Model Evaluation

Building a model is only half the job. You need to evaluate it properly.

## Key Concepts

### Classification Metrics
- **Accuracy**: Correct / Total (can be misleading!)
- **Precision**: TP / (TP + FP) - "Of positive predictions, how many correct?"
- **Recall**: TP / (TP + FN) - "Of actual positives, how many found?"
- **F1 Score**: Harmonic mean of precision and recall

### Regression Metrics
- **MSE**: Mean Squared Error
- **RMSE**: Root MSE (same units as target)
- **MAE**: Mean Absolute Error (robust to outliers)
- **R²**: Proportion of variance explained

### Cross-Validation
- **Why**: Single train/test split is unreliable
- **K-Fold**: Split data into K parts, rotate test set
- **Stratified**: Maintain class distribution in folds

### Interview Essentials
- Know when to use each metric
- Implement cross-validation from scratch
- Discuss precision-recall trade-offs

Let's practice evaluation!
    `,problems:Le("model-evaluation",["precision-recall-f1","cross-validation","confusion-matrix"])},{id:"neural-networks",title:"Neural Network Fundamentals",description:"Perceptrons to MLPs: activations, forward and backward passes, initialization and normalization.",icon:"🧠",introduction:`
# Deep Learning Basics

Core building blocks that appear in every neural network.

## Key Concepts

### Activation Functions
- **ReLU**: max(0, x) - Most common, avoids vanishing gradient
- **Sigmoid**: 1/(1+e^-x) - Output in [0,1], used for binary classification
- **Softmax**: Converts logits to probabilities that sum to 1
- **Tanh**: Output in [-1, 1], zero-centered

### When to Use Each
- **Hidden layers**: ReLU (or variants like LeakyReLU)
- **Binary output**: Sigmoid
- **Multi-class output**: Softmax
- **RNNs**: Tanh (historically)

### Dense Layers
- **Forward pass**: output = activation(W @ x + b)
- **Parameters**: Weights W and biases b
- **Computation**: Matrix multiplication + bias + activation

Let's implement these fundamentals!

---

## Neural Networks

Neural networks are the foundation of deep learning. Understanding the math behind them is essential for ML interviews.

### Key Concepts

#### Multi-Layer Perceptron (MLP)
- **Architecture**: Input → Hidden layers → Output
- **Forward pass**: Compute activations layer by layer
- **Activation functions**: ReLU, Sigmoid, Tanh, Softmax

#### Backpropagation
- **Chain rule**: Compute gradients layer by layer
- **Weight updates**: Gradient descent on all parameters
- **Key insight**: Errors propagate backwards through the network

#### Training Techniques
- **Weight Initialization**: Xavier, He initialization
- **Batch Normalization**: Normalize activations for stable training
- **Dropout**: Randomly drop neurons to prevent overfitting

#### Loss Functions
- **Cross-entropy**: For classification
- **MSE**: For regression

#### Common Interview Topics
- Implement forward/backward pass from scratch
- Explain vanishing/exploding gradients
- Discuss initialization strategies
- Implement regularization techniques

Let's build neural networks from scratch!
    `,problems:Le("neural-networks",["perceptron","neural-network-forward","backpropagation","cross-entropy-loss","mlp-forward-backward","weight-init","batch-norm","dropout"])},{id:"optimization",title:"Optimization & Training",description:"Optimizers, learning-rate schedules and the tricks that make training stable.",icon:"⚙️",introduction:`
# Optimization & Training

_This section is being written._
    `,problems:Le("optimization",[])},{id:"cnn",title:"Vision: CNNs & ViTs",description:"Convolutions, pooling, residual blocks and patch embeddings for images.",icon:"🖼️",introduction:`
# Convolutional Neural Networks (CNNs)

CNNs are the backbone of computer vision. They learn hierarchical features from images.

## Key Concepts

### Convolution Operation
- **Kernel/Filter**: Small matrix that slides over input
- **Feature maps**: Output of applying filters
- **Parameters**: Stride, padding, kernel size

### Pooling Layers
- **Max pooling**: Take maximum in each region
- **Average pooling**: Take average in each region
- **Purpose**: Reduce spatial dimensions, add translation invariance

### CNN Architecture Components
1. **Convolutional layers**: Extract features
2. **Pooling layers**: Reduce dimensions
3. **Flatten**: Convert 2D to 1D
4. **Fully connected**: Classification

### Output Size Formula
\`\`\`
output = (input - kernel + 2*padding) / stride + 1
\`\`\`

### Classic Architectures
- **LeNet**: First successful CNN
- **AlexNet**: Deep CNNs for ImageNet
- **VGG**: Very deep, small kernels
- **ResNet**: Skip connections

Let's implement CNN operations!
    `,problems:Le("cnn",["conv-output-size","conv2d-forward","max-pool","flatten-layer","conv2d-advanced"])},{id:"sequence-models",title:"Sequence Models",description:"RNNs, LSTMs and GRUs: modelling sequences before (and alongside) attention.",icon:"🔁",introduction:`
# Sequence Models

_This section is being written._
    `,problems:Le("sequence-models",[])},{id:"transformers",title:"Attention & Transformers",description:"Master attention mechanisms and transformer architecture.",icon:"🤖",introduction:`
# Attention and Transformers

Transformers have revolutionized NLP and are now used in vision, speech, and more.

## Key Concepts

### Self-Attention
- **Query, Key, Value**: Three projections of input
- **Attention scores**: Q @ K.T / sqrt(d_k)
- **Output**: Weighted sum of values

### Scaled Dot-Product Attention
\`\`\`
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
\`\`\`

### Multi-Head Attention
- **Multiple heads**: Parallel attention with different projections
- **Concatenate**: Combine all head outputs
- **Project**: Linear transformation back to model dimension

### Transformer Components
- **Positional Encoding**: Inject position information
- **Layer Normalization**: Normalize across features
- **Feed-Forward Network**: MLP after attention
- **Residual Connections**: Add input to output

### Causal Masking
- **Purpose**: Prevent attending to future tokens
- **Implementation**: Add -inf to upper triangle before softmax

### Interview Essentials
- Implement scaled dot-product attention
- Explain why we scale by sqrt(d_k)
- Multi-head attention benefits
- Positional encoding purpose

Let's build transformers from scratch!
    `,problems:Le("transformers",["scaled-dot-product-attention","multi-head-attention","positional-encoding","layer-norm","causal-mask"])},{id:"llm-internals",title:"LLM Internals",description:"Tokenizers, sampling, KV caches, RoPE, GQA, MoE: how modern language models actually run.",icon:"🧩",introduction:`
# LLM Internals

_This section is being written._
    `,problems:Le("llm-internals",[])},{id:"representation-learning",title:"Representation Learning & Retrieval",description:"Contrastive learning, embedding search and retrieval for RAG.",icon:"🧭",introduction:`
# Representation Learning & Retrieval

_This section is being written._
    `,problems:Le("representation-learning",[])},{id:"generative-models",title:"Generative Models",description:"Learn VAEs, diffusion models, and generative AI fundamentals.",icon:"🎨",introduction:`
# Generative Models

Generative models learn to create new data similar to the training distribution.

## Key Concepts

### Variational Autoencoders (VAEs)
- **Encoder**: Maps input to latent distribution (μ, σ)
- **Decoder**: Reconstructs input from latent sample
- **Reparameterization trick**: Enable backprop through sampling
- **Loss**: Reconstruction + KL divergence

### VAE Loss (ELBO)
\`\`\`
L = Reconstruction Loss + KL Divergence
L = ||x - x_reconstructed||² + KL(q(z|x) || p(z))
\`\`\`

### Diffusion Models
- **Forward process**: Gradually add noise to data
- **Reverse process**: Learn to denoise
- **Noise schedule**: β_t controls noise at each step

### Key Formulas
\`\`\`
x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
\`\`\`

### KL Divergence
- Measures difference between distributions
- KL(P || Q) ≥ 0, equals 0 iff P = Q
- Not symmetric

### Interview Topics
- Explain reparameterization trick
- VAE loss components
- Diffusion forward/reverse process
- Compare GANs vs VAEs vs Diffusion

Let's implement generative models!
    `,problems:Le("generative-models",["kl-divergence","vae-reparameterization","vae-loss","vqvae-quantization","diffusion-noise-schedule","diffusion-forward"])},{id:"reinforcement-learning",title:"Reinforcement Learning",description:"Master RL fundamentals from Q-learning to policy gradients.",icon:"🎮",introduction:`
# Reinforcement Learning

Reinforcement learning (RL) trains agents to make decisions by learning from rewards and punishments through trial and error.

## Key Concepts

### The RL Framework
- **Agent**: The learner/decision-maker
- **Environment**: What the agent interacts with
- **State (s)**: Current situation
- **Action (a)**: What the agent can do
- **Reward (r)**: Feedback signal
- **Policy (π)**: Strategy for choosing actions

### Value Functions
- **State Value V(s)**: Expected return starting from state s
- **Action Value Q(s,a)**: Expected return taking action a in state s

### Bellman Equations
\`\`\`python
V(s) = E[r + γ * V(s')]
Q(s,a) = E[r + γ * max_a' Q(s',a')]
\`\`\`

### Key Algorithms

| Algorithm | Type | Policy |
|-----------|------|--------|
| Q-Learning | Value-based | Off-policy |
| SARSA | Value-based | On-policy |
| REINFORCE | Policy-based | On-policy |
| Actor-Critic | Hybrid | On-policy |
| PPO | Policy-based | On-policy |

### Exploration vs Exploitation
- **Exploration**: Try new actions to discover better strategies
- **Exploitation**: Use known good actions to maximize reward
- **ε-greedy**: Simple balance (random with probability ε)

### RL for LLMs (RLHF)
- **Reward Modeling**: Learn preferences from human feedback
- **PPO for LLMs**: Fine-tune with KL constraints
- **DPO**: Direct preference optimization without reward model

### Interview Essentials
- Implement Q-learning and SARSA from scratch
- Explain on-policy vs off-policy
- Understand policy gradients and REINFORCE
- Know GAE and PPO for modern deep RL
- **RLHF** (reward models, DPO, KL penalties) lives in the **Post-training & Efficiency** section, which builds on these foundations

Master these concepts for ML interviews at top companies!
    `,problems:Le("reinforcement-learning",["rl-discounted-return","rl-epsilon-greedy","rl-bellman-value","rl-q-learning-update","rl-sarsa-update","rl-td0-prediction","rl-value-iteration","rl-n-step-return","rl-policy-gradient","rl-advantage-estimation","rl-ppo-clip"])},{id:"post-training",title:"Post-training & Efficiency",description:"Reward models, DPO, GRPO, LoRA and quantization: turning a base model into a useful, cheap one.",icon:"🎯",introduction:`
# Post-training & Efficiency

A pretrained language model predicts text. Post-training turns it into an assistant that follows instructions and reflects human preferences, and efficiency techniques make it affordable to train and serve.

## Preference learning
| Method | Idea | Loss |
|--------|------|------|
| Reward model | Learn a scalar score from pairwise human preferences (Bradley–Terry) | \`-log σ(r_chosen - r_rejected)\` |
| PPO-RLHF | Maximize reward with a KL penalty that keeps the policy near the reference model | clipped surrogate with \`A = r - β·KL\` |
| DPO | Skip the reward model: optimize preferences directly through log-probability ratios | \`-log σ(β·(Δ_chosen - Δ_rejected))\` |

## Why the KL penalty?
Without it, the policy drifts toward outputs that fool the reward model ("reward hacking") and forgets how to write fluent text. The KL term measures how far the policy has moved from the frozen reference model.

## Efficiency
Fine-tuning and serving large models is dominated by memory. Low-rank adapters (LoRA) train a tiny fraction of the weights, and quantization stores weights in fewer bits.
    `,problems:Le("post-training",["rl-reward-modeling","rl-kl-penalty","rl-ppo-llm-objective","rl-dpo-loss"])},{id:"e2e-implementations",title:"End-to-End Implementations",description:"Build complete ML models from scratch.",icon:"🏗️",introduction:`
# End-to-End Implementations

Put everything together to build complete models from scratch.

## Why Build From Scratch?

1. **Deep Understanding**: Know every component
2. **Interview Preparation**: Common coding questions
3. **Debugging Skills**: Understand what can go wrong
4. **Framework Independence**: Adapt to any library

## Models Covered

### 2-Layer MLP with Backprop
Complete neural network with:
- Forward pass through hidden layer
- Backpropagation of gradients
- Weight updates via gradient descent

### Transformer Encoder
Full encoder block with:
- Multi-head self-attention
- Positional encoding
- Layer normalization
- Feed-forward network
- Residual connections

### Variational Autoencoder (VAE)
Generative model with:
- Encoder → latent distribution
- Reparameterization trick
- Decoder → reconstruction
- ELBO loss (reconstruction + KL)

### Diffusion Model
DDPM-style diffusion with:
- Noise schedule (β, α, ᾱ)
- Forward diffusion process
- Reverse denoising process
- Training objective

### Convolutional Neural Network
Image classifier with:
- Conv2D layers
- Max pooling
- Flatten + FC layers
- Full forward pass

Build these models to truly understand deep learning!
    `,problems:Le("e2e-implementations",["e2e-mlp","e2e-transformer","e2e-vae","e2e-vqvae","e2e-diffusion","e2e-cnn"])}],Cn=wb.filter(e=>e.problems.length>0);function Tb({isOpen:e,onClose:t}){const[n,r]=C.useState(""),[i,o]=C.useState(""),[a,s]=C.useState(!1),[l,u]=C.useState(!1),[c,d]=C.useState("");if(!e)return null;const h=async f=>{f.preventDefault(),s(!0),d("");try{(await fetch("https://formspree.io/f/mpqqwpav",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:n,feedback:i,_subject:`ML Coding Lab Feedback from ${n}`})})).ok?(u(!0),r(""),o("")):d("Failed to send feedback. Please try again.")}catch{d("Failed to send feedback. Please try again.")}finally{s(!1)}},m=()=>{u(!1),d(""),t()};return p.jsxs("div",{className:"fixed inset-0 z-50 flex items-center justify-center",children:[p.jsx("div",{className:"absolute inset-0 bg-black bg-opacity-50",onClick:m}),p.jsxs("div",{className:"relative bg-white rounded-lg shadow-xl w-full max-w-md mx-4 p-6",children:[p.jsx("button",{onClick:m,className:"absolute top-4 right-4 text-gray-400 hover:text-gray-600",children:p.jsx("svg",{className:"w-5 h-5",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M6 18L18 6M6 6l12 12"})})}),p.jsx("h2",{className:"text-xl font-semibold text-gray-900 mb-4",children:"Send Feedback"}),l?p.jsxs("div",{className:"text-center py-8",children:[p.jsx("div",{className:"w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4",children:p.jsx("svg",{className:"w-6 h-6 text-green-600",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M5 13l4 4L19 7"})})}),p.jsx("p",{className:"text-gray-700 mb-4",children:"Thank you for your feedback!"}),p.jsx("button",{onClick:m,className:"px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 transition-colors",children:"Close"})]}):p.jsxs("form",{onSubmit:h,className:"space-y-4",children:[p.jsxs("div",{children:[p.jsx("label",{htmlFor:"name",className:"block text-sm font-medium text-gray-700 mb-1",children:"Name"}),p.jsx("input",{type:"text",id:"name",value:n,onChange:f=>r(f.target.value),required:!0,className:"w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent",placeholder:"Your name"})]}),p.jsxs("div",{children:[p.jsx("label",{htmlFor:"feedback",className:"block text-sm font-medium text-gray-700 mb-1",children:"Feedback"}),p.jsx("textarea",{id:"feedback",value:i,onChange:f=>o(f.target.value),required:!0,rows:4,className:"w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none",placeholder:"Share your thoughts, suggestions, or report issues..."})]}),c&&p.jsx("p",{className:"text-red-600 text-sm",children:c}),p.jsxs("div",{className:"flex gap-3 justify-end",children:[p.jsx("button",{type:"button",onClick:m,className:"px-4 py-2 text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors",children:"Cancel"}),p.jsx("button",{type:"submit",disabled:a,className:"px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:a?"Sending...":"Send Feedback"})]})]})]})]})}function Td(){const[e,t]=C.useState(()=>{const r=localStorage.getItem("ml-lab-dark-mode");return r!==null?r==="true":!0});return C.useEffect(()=>{const r=document.documentElement;e?r.classList.add("dark"):r.classList.remove("dark"),localStorage.setItem("ml-lab-dark-mode",String(e))},[e]),{isDark:e,toggle:()=>t(r=>!r)}}function Cb({onMenuToggle:e,showMenuOnDesktop:t=!1}){const{getOverallProgress:n}=lo(),[r,i]=C.useState(!1),{isDark:o,toggle:a}=Td(),s=Cn.map(u=>({id:u.id,problemCount:u.problems.length})),l=n(s);return p.jsxs(p.Fragment,{children:[p.jsx("header",{className:"bg-white/80 dark:bg-dark-900/80 backdrop-blur-xl backdrop-saturate-[1.2] border-b border-gray-200 dark:border-dark-500 sticky top-0 z-50",children:p.jsx("div",{className:"max-w-7xl mx-auto px-4 sm:px-6 lg:px-8",children:p.jsxs("div",{className:"flex items-center justify-between h-16",children:[p.jsxs("div",{className:"flex items-center gap-3",children:[e&&p.jsx("button",{onClick:e,className:`${t?"":"lg:hidden"} p-2 -ml-2 text-gray-600 dark:text-dark-200 hover:text-gray-900 dark:hover:text-dark-100 hover:bg-gray-100 dark:hover:bg-dark-600 rounded-lg transition-colors`,"aria-label":"Toggle menu",children:p.jsx("svg",{className:"w-5 h-5",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M4 6h16M4 12h16M4 18h16"})})}),p.jsxs(Ye,{to:"/",className:"flex items-center gap-3",children:[p.jsx("div",{className:"w-8 h-8 bg-gradient-to-br from-primary-500 to-accent-500 rounded-lg flex items-center justify-center font-mono font-bold text-white text-xs",children:"ML"}),p.jsx("span",{className:"text-lg font-semibold text-gray-900 dark:text-dark-100 whitespace-nowrap tracking-tight",children:"ML Coding Lab"})]})]}),p.jsxs("div",{className:"flex items-center gap-2",children:[p.jsxs(Ye,{to:"/scratchpad",className:"hidden lg:flex px-3 py-1.5 text-gray-600 dark:text-dark-200 text-sm font-medium rounded-lg hover:bg-gray-100 dark:hover:bg-dark-800 hover:text-gray-900 dark:hover:text-dark-100 border border-transparent hover:border-gray-200 dark:hover:border-dark-500 transition-all items-center gap-2",children:[p.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"})}),"Scratchpad"]}),p.jsxs("a",{href:"/ml-coding-lab/ml-cheatsheet.html",target:"_blank",rel:"noopener noreferrer",className:"hidden lg:flex px-3 py-1.5 text-primary-400 dark:text-primary-300 text-sm font-medium rounded-lg bg-primary-50 dark:bg-primary-500/10 border border-primary-200 dark:border-primary-500/20 hover:bg-primary-100 dark:hover:bg-primary-500/20 hover:border-primary-300 dark:hover:border-primary-500/30 transition-all items-center gap-2",children:[p.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"})}),"Cheat Sheet"]}),p.jsxs("button",{onClick:()=>i(!0),className:"hidden lg:flex px-3 py-1.5 text-gray-500 dark:text-dark-200 text-sm font-medium hover:text-gray-700 dark:hover:text-dark-100 transition-colors items-center gap-1",children:[p.jsx("svg",{className:"w-4 h-4",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"})}),"Feedback"]}),p.jsx("button",{onClick:a,className:"p-2 text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100 hover:bg-gray-100 dark:hover:bg-dark-600 rounded-lg transition-colors","aria-label":"Toggle dark mode",children:o?p.jsx("svg",{className:"w-5 h-5",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"})}):p.jsx("svg",{className:"w-5 h-5",fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"})})}),p.jsxs("div",{className:"hidden xl:flex items-center gap-2.5 px-3.5 py-1.5 rounded-full bg-gray-100 dark:bg-dark-800 border border-gray-200 dark:border-dark-500 text-xs",children:[p.jsx("div",{className:"w-20 h-1 bg-gray-200 dark:bg-dark-500 rounded-full overflow-hidden",children:p.jsx("div",{className:"h-full progress-gradient rounded-full transition-all duration-500",style:{width:`${l}%`}})}),p.jsxs("span",{className:"font-mono font-semibold text-primary-600 dark:text-primary-300",children:[l,"%"]})]})]})]})})}),p.jsx(Tb,{isOpen:r,onClose:()=>i(!1)})]})}function mp({onNavigate:e}){const{getSectionProgress:t}=lo();return p.jsx("aside",{className:"w-64 bg-white dark:bg-dark-800/50 border-r border-gray-200 dark:border-dark-500 min-h-[calc(100vh-4rem)] p-4",children:p.jsxs("nav",{className:"space-y-1",children:[p.jsx("h3",{className:"text-dark-300 dark:text-dark-300 text-[11px] font-semibold uppercase tracking-widest mb-4 px-3 font-mono",children:"Curriculum"}),Cn.map((n,r)=>{const i=t(n.id,n.problems.length);return p.jsxs(rx,{to:`/section/${n.id}`,onClick:e,className:({isActive:o})=>`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${o?"bg-primary-50 dark:bg-primary-500/10 text-primary-700 dark:text-primary-300 border border-primary-200 dark:border-primary-500/20":"text-gray-700 dark:text-dark-200 hover:bg-gray-100 dark:hover:bg-dark-600 hover:text-gray-900 dark:hover:text-dark-100 border border-transparent"}`,children:[p.jsx("span",{className:"w-6 h-6 flex items-center justify-center text-lg",children:n.icon}),p.jsxs("div",{className:"flex-1 min-w-0",children:[p.jsxs("div",{className:"flex items-center gap-2",children:[p.jsx("span",{className:"text-dark-300 text-[10px] font-mono",children:String(r+1).padStart(2,"0")}),p.jsx("span",{className:"text-sm font-medium truncate",children:n.title})]}),p.jsxs("div",{className:"flex items-center gap-2 mt-1",children:[p.jsx("div",{className:"flex-1 h-0.5 bg-gray-200 dark:bg-dark-500 rounded-full overflow-hidden",children:p.jsx("div",{className:"h-full progress-gradient rounded-full transition-all duration-300",style:{width:`${i}%`}})}),p.jsxs("span",{className:"text-[10px] text-dark-300 font-mono",children:[i,"%"]})]})]})]},n.id)})]})})}function Sb(){const e=new Date().getFullYear();return p.jsx("footer",{className:"border-t border-gray-200 dark:border-dark-500 mt-auto",children:p.jsxs("div",{className:"max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6",children:[p.jsxs("div",{className:"flex flex-col md:flex-row justify-between items-center gap-4",children:[p.jsxs("div",{className:"text-gray-500 dark:text-dark-300 text-xs",children:["© ",e," Siddharth Choudhary. All rights reserved."]}),p.jsxs("div",{className:"flex items-center gap-6 text-xs",children:[p.jsx(Ye,{to:"/terms",className:"text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100 transition-colors",children:"Terms"}),p.jsx(Ye,{to:"/privacy",className:"text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100 transition-colors",children:"Privacy"}),p.jsxs("a",{href:"https://github.com/itzsid/ml-coding-lab",target:"_blank",rel:"noopener noreferrer",className:"text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100 transition-colors flex items-center gap-1",children:[p.jsx("svg",{className:"w-3.5 h-3.5",fill:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{fillRule:"evenodd",clipRule:"evenodd",d:"M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.604-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"})}),"GitHub"]})]})]}),p.jsx("div",{className:"mt-4 pt-4 border-t border-gray-100 dark:border-dark-500/50 text-center text-xs text-gray-400 dark:text-dark-300",children:p.jsx("p",{children:"Code licensed under MIT. Educational content is copyrighted."})})]})})}function zb({children:e}){const[t,n]=C.useState(!1),{pathname:r}=ur(),i=r.startsWith("/problem/")||r==="/scratchpad";return p.jsxs("div",{className:"min-h-screen bg-gray-50 dark:bg-dark-900 flex flex-col",children:[p.jsx("div",{className:"bg-mesh"}),p.jsx("div",{className:"bg-noise"}),p.jsxs("div",{className:"relative z-[1] flex flex-col min-h-screen",children:[p.jsx(Cb,{onMenuToggle:()=>n(!t),showMenuOnDesktop:i}),p.jsxs("div",{className:"flex flex-1",children:[!i&&p.jsx("div",{className:"hidden lg:block",children:p.jsx(mp,{})}),t&&p.jsxs("div",{className:`fixed inset-0 z-40 ${i?"":"lg:hidden"}`,children:[p.jsx("div",{className:"fixed inset-0 bg-black/50 backdrop-blur-sm",onClick:()=>n(!1)}),p.jsx("div",{className:"fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-dark-800 shadow-xl overflow-y-auto border-r border-gray-200 dark:border-dark-500",children:p.jsx("div",{className:"pt-2",children:p.jsx(mp,{onNavigate:()=>n(!1)})})})]}),p.jsx("main",{className:"flex-1 min-w-0 p-6",children:e})]}),!i&&p.jsx(Sb,{})]})]})}const fp="https://itzsid.github.io/ml-coding-lab",Eb="ML Coding Lab - Learn Machine Learning by Building",Pb="Practice hands-on machine learning coding problems. Implement NumPy, neural networks, CNNs, transformers, and more. Run Python code directly in your browser with instant feedback.",Nb="machine learning, ML, deep learning, neural networks, Python, NumPy, transformers, CNN, PyTorch, coding practice, interview prep, AI";function ei({title:e,description:t=Pb,keywords:n=Nb,canonical:r,type:i="website"}){const o=e?`${e} | ML Coding Lab`:Eb,a=r?`${fp}${r}`:fp;return p.jsxs(Fx,{children:[p.jsx("title",{children:o}),p.jsx("meta",{name:"title",content:o}),p.jsx("meta",{name:"description",content:t}),p.jsx("meta",{name:"keywords",content:n}),p.jsx("link",{rel:"canonical",href:a}),p.jsx("meta",{property:"og:type",content:i}),p.jsx("meta",{property:"og:url",content:a}),p.jsx("meta",{property:"og:title",content:o}),p.jsx("meta",{property:"og:description",content:t}),p.jsx("meta",{property:"twitter:url",content:a}),p.jsx("meta",{property:"twitter:title",content:o}),p.jsx("meta",{property:"twitter:description",content:t})]})}function Ab(){const{getSectionProgress:e,getOverallProgress:t}=lo(),n=Cn.map(o=>({id:o.id,problemCount:o.problems.length})),r=t(n),i=C.useMemo(()=>Cn.reduce((o,a)=>o+a.problems.length,0),[]);return p.jsxs("div",{className:"max-w-[1200px] mx-auto",children:[p.jsx(ei,{canonical:"/"}),p.jsxs("div",{className:"text-center mb-16 pt-8",children:[p.jsxs("div",{className:"inline-flex items-center gap-2 px-4 py-1.5 bg-white dark:bg-dark-800 border border-gray-200 dark:border-dark-500 rounded-full text-sm text-gray-600 dark:text-dark-200 mb-8 animate-fade-up",children:[p.jsx("span",{className:"w-2 h-2 rounded-full bg-accent-500 animate-pulse-dot"}),"100% browser-based · No setup required"]}),p.jsxs("h1",{className:"font-display text-[clamp(2.5rem,6vw,4.2rem)] font-normal leading-[1.1] tracking-tight text-gray-900 dark:text-dark-100 mb-6 animate-fade-up [animation-delay:0.1s]",children:["Learn Machine Learning",p.jsx("br",{}),"by ",p.jsx("em",{className:"text-primary-400 dark:text-primary-300",children:"Building"})]}),p.jsx("p",{className:"text-lg text-gray-500 dark:text-dark-200 max-w-[560px] mx-auto mb-10 font-light leading-relaxed animate-fade-up [animation-delay:0.2s]",children:"Implement algorithms from scratch — NumPy, neural networks, transformers, diffusion models. Write real Python with instant feedback, directly in your browser."}),p.jsxs("div",{className:"flex justify-center gap-12 animate-fade-up [animation-delay:0.3s]",children:[p.jsxs("div",{className:"text-center",children:[p.jsxs("div",{className:"font-mono text-2xl font-bold text-gray-900 dark:text-dark-100",children:[i,"+"]}),p.jsx("div",{className:"text-[11px] text-gray-400 dark:text-dark-300 uppercase tracking-widest mt-1",children:"Problems"})]}),p.jsxs("div",{className:"text-center",children:[p.jsx("div",{className:"font-mono text-2xl font-bold text-gray-900 dark:text-dark-100",children:Cn.length}),p.jsx("div",{className:"text-[11px] text-gray-400 dark:text-dark-300 uppercase tracking-widest mt-1",children:"Sections"})]}),p.jsxs("div",{className:"text-center",children:[p.jsxs("div",{className:"font-mono text-2xl font-bold text-gray-900 dark:text-dark-100",children:[r,"%"]}),p.jsx("div",{className:"text-[11px] text-gray-400 dark:text-dark-300 uppercase tracking-widest mt-1",children:"Complete"})]})]})]}),p.jsx("div",{className:"text-[11px] font-semibold uppercase tracking-[0.12em] text-gray-400 dark:text-dark-300 mb-6 font-mono",children:"Curriculum"}),p.jsx("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-4",children:Cn.map((o,a)=>{const s=e(o.id,o.problems.length);return p.jsxs(Ye,{to:`/section/${o.id}`,className:"card-stagger animate-fade-up group relative flex gap-4 p-5 bg-white dark:bg-dark-800 rounded-[14px] border border-gray-200 dark:border-dark-500 hover:border-gray-300 dark:hover:border-dark-400 hover:bg-gray-50 dark:hover:bg-dark-700 hover:-translate-y-0.5 hover:shadow-lg dark:hover:shadow-[0_8px_32px_rgba(0,0,0,0.2)] transition-all duration-250 overflow-hidden",children:[p.jsx("div",{className:"absolute inset-0 rounded-[14px] bg-gradient-to-br from-primary-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"}),p.jsx("div",{className:"shrink-0 w-11 h-11 bg-gray-100 dark:bg-dark-600 border border-gray-200 dark:border-dark-500 rounded-[10px] flex items-center justify-center text-[22px] group-hover:bg-primary-50 dark:group-hover:bg-primary-500/10 group-hover:border-primary-200 dark:group-hover:border-primary-500/25 transition-all relative z-[1]",children:o.icon}),p.jsxs("div",{className:"flex-1 min-w-0 relative z-[1]",children:[p.jsx("div",{className:"font-mono text-[11px] text-gray-400 dark:text-dark-300 mb-0.5",children:String(a+1).padStart(2,"0")}),p.jsx("div",{className:"text-base font-semibold text-gray-900 dark:text-dark-100 tracking-tight mb-1.5 group-hover:text-primary-600 dark:group-hover:text-primary-300 transition-colors",children:o.title}),p.jsx("div",{className:"text-[13px] text-gray-500 dark:text-dark-300 leading-relaxed mb-3.5",children:o.description}),p.jsxs("div",{className:"flex items-center justify-between",children:[p.jsxs("span",{className:"font-mono text-[11px] text-gray-400 dark:text-dark-300",children:[o.problems.length," problems"]}),p.jsxs("div",{className:"flex items-center gap-2",children:[p.jsx("div",{className:"w-[60px] h-[3px] bg-gray-200 dark:bg-dark-500 rounded-full overflow-hidden",children:p.jsx("div",{className:"h-full progress-gradient rounded-full transition-all duration-500",style:{width:`${s}%`}})}),p.jsxs("span",{className:"font-mono text-[11px] text-gray-400 dark:text-dark-300",children:[s,"%"]})]})]})]}),p.jsx("span",{className:"absolute right-5 top-1/2 -translate-y-1/2 -translate-x-1 opacity-0 text-primary-500 dark:text-primary-300 text-lg group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-250",children:"→"})]},o.id)})}),p.jsx("div",{className:"mt-16 mb-8",children:p.jsxs("div",{className:"grid grid-cols-1 md:grid-cols-3 gap-px bg-gray-200 dark:bg-dark-500 border border-gray-200 dark:border-dark-500 rounded-[14px] overflow-hidden",children:[p.jsxs("div",{className:"bg-white dark:bg-dark-800 p-8 text-center hover:bg-gray-50 dark:hover:bg-dark-700 transition-colors",children:[p.jsx("div",{className:"w-12 h-12 mx-auto mb-4 bg-gray-100 dark:bg-dark-600 border border-gray-200 dark:border-dark-500 rounded-xl flex items-center justify-center text-[22px]",children:"🐍"}),p.jsx("div",{className:"font-semibold text-sm text-gray-900 dark:text-dark-100 tracking-tight mb-2",children:"Python in Browser"}),p.jsx("div",{className:"text-[13px] text-gray-500 dark:text-dark-300 leading-relaxed",children:"Full Python runtime with NumPy, running locally via WebAssembly. No install, no config."})]}),p.jsxs("div",{className:"bg-white dark:bg-dark-800 p-8 text-center hover:bg-gray-50 dark:hover:bg-dark-700 transition-colors",children:[p.jsx("div",{className:"w-12 h-12 mx-auto mb-4 bg-gray-100 dark:bg-dark-600 border border-gray-200 dark:border-dark-500 rounded-xl flex items-center justify-center text-[22px]",children:"✅"}),p.jsx("div",{className:"font-semibold text-sm text-gray-900 dark:text-dark-100 tracking-tight mb-2",children:"Instant Feedback"}),p.jsx("div",{className:"text-[13px] text-gray-500 dark:text-dark-300 leading-relaxed",children:"Run your code against test suites and see results immediately."})]}),p.jsxs("div",{className:"bg-white dark:bg-dark-800 p-8 text-center hover:bg-gray-50 dark:hover:bg-dark-700 transition-colors",children:[p.jsx("div",{className:"w-12 h-12 mx-auto mb-4 bg-gray-100 dark:bg-dark-600 border border-gray-200 dark:border-dark-500 rounded-xl flex items-center justify-center text-[22px]",children:"💾"}),p.jsx("div",{className:"font-semibold text-sm text-gray-900 dark:text-dark-100 tracking-tight mb-2",children:"Progress Saved Locally"}),p.jsx("div",{className:"text-[13px] text-gray-500 dark:text-dark-300 leading-relaxed",children:"Your code and progress persist in your browser. Pick up anytime."})]})]})})]})}const gp={"python-basics":"numpy-fundamentals","deep-learning":"neural-networks"};function Lb(e,t){const n={};return(e[e.length-1]===""?[...e,""]:e).join((n.padRight?" ":"")+","+(n.padLeft===!1?"":" ")).trim()}const jb=/^[$_\p{ID_Start}][$_\u{200C}\u{200D}\p{ID_Continue}]*$/u,Rb=/^[$_\p{ID_Start}][-$_\u{200C}\u{200D}\p{ID_Continue}]*$/u,Fb={};function _p(e,t){return(Fb.jsx?Rb:jb).test(e)}const Ib=/[ \t\n\f\r]/g;function Db(e){return typeof e=="object"?e.type==="text"?yp(e.value):!1:yp(e)}function yp(e){return e.replace(Ib,"")===""}class uo{constructor(t,n,r){this.normal=n,this.property=t,r&&(this.space=r)}}uo.prototype.normal={};uo.prototype.property={};uo.prototype.space=void 0;function s0(e,t){const n={},r={};for(const i of e)Object.assign(n,i.property),Object.assign(r,i.normal);return new uo(n,r,t)}function hu(e){return e.toLowerCase()}class lt{constructor(t,n){this.attribute=n,this.property=t}}lt.prototype.attribute="";lt.prototype.booleanish=!1;lt.prototype.boolean=!1;lt.prototype.commaOrSpaceSeparated=!1;lt.prototype.commaSeparated=!1;lt.prototype.defined=!1;lt.prototype.mustUseProperty=!1;lt.prototype.number=!1;lt.prototype.overloadedBoolean=!1;lt.prototype.property="";lt.prototype.spaceSeparated=!1;lt.prototype.space=void 0;let Mb=0;const J=dr(),ze=dr(),mu=dr(),j=dr(),he=dr(),Mr=dr(),ut=dr();function dr(){return 2**++Mb}const fu=Object.freeze(Object.defineProperty({__proto__:null,boolean:J,booleanish:ze,commaOrSpaceSeparated:ut,commaSeparated:Mr,number:j,overloadedBoolean:mu,spaceSeparated:he},Symbol.toStringTag,{value:"Module"})),Hs=Object.keys(fu);class Cd extends lt{constructor(t,n,r,i){let o=-1;if(super(t,n),xp(this,"space",i),typeof r=="number")for(;++o<Hs.length;){const a=Hs[o];xp(this,Hs[o],(r&fu[a])===fu[a])}}}Cd.prototype.defined=!0;function xp(e,t,n){n&&(e[t]=n)}function ti(e){const t={},n={};for(const[r,i]of Object.entries(e.properties)){const o=new Cd(r,e.transform(e.attributes||{},r),i,e.space);e.mustUseProperty&&e.mustUseProperty.includes(r)&&(o.mustUseProperty=!0),t[r]=o,n[hu(r)]=r,n[hu(o.attribute)]=r}return new uo(t,n,e.space)}const l0=ti({properties:{ariaActiveDescendant:null,ariaAtomic:ze,ariaAutoComplete:null,ariaBusy:ze,ariaChecked:ze,ariaColCount:j,ariaColIndex:j,ariaColSpan:j,ariaControls:he,ariaCurrent:null,ariaDescribedBy:he,ariaDetails:null,ariaDisabled:ze,ariaDropEffect:he,ariaErrorMessage:null,ariaExpanded:ze,ariaFlowTo:he,ariaGrabbed:ze,ariaHasPopup:null,ariaHidden:ze,ariaInvalid:null,ariaKeyShortcuts:null,ariaLabel:null,ariaLabelledBy:he,ariaLevel:j,ariaLive:null,ariaModal:ze,ariaMultiLine:ze,ariaMultiSelectable:ze,ariaOrientation:null,ariaOwns:he,ariaPlaceholder:null,ariaPosInSet:j,ariaPressed:ze,ariaReadOnly:ze,ariaRelevant:null,ariaRequired:ze,ariaRoleDescription:he,ariaRowCount:j,ariaRowIndex:j,ariaRowSpan:j,ariaSelected:ze,ariaSetSize:j,ariaSort:null,ariaValueMax:j,ariaValueMin:j,ariaValueNow:j,ariaValueText:null,role:null},transform(e,t){return t==="role"?t:"aria-"+t.slice(4).toLowerCase()}});function u0(e,t){return t in e?e[t]:t}function d0(e,t){return u0(e,t.toLowerCase())}const qb=ti({attributes:{acceptcharset:"accept-charset",classname:"class",htmlfor:"for",httpequiv:"http-equiv"},mustUseProperty:["checked","multiple","muted","selected"],properties:{abbr:null,accept:Mr,acceptCharset:he,accessKey:he,action:null,allow:null,allowFullScreen:J,allowPaymentRequest:J,allowUserMedia:J,alt:null,as:null,async:J,autoCapitalize:null,autoComplete:he,autoFocus:J,autoPlay:J,blocking:he,capture:null,charSet:null,checked:J,cite:null,className:he,cols:j,colSpan:null,content:null,contentEditable:ze,controls:J,controlsList:he,coords:j|Mr,crossOrigin:null,data:null,dateTime:null,decoding:null,default:J,defer:J,dir:null,dirName:null,disabled:J,download:mu,draggable:ze,encType:null,enterKeyHint:null,fetchPriority:null,form:null,formAction:null,formEncType:null,formMethod:null,formNoValidate:J,formTarget:null,headers:he,height:j,hidden:mu,high:j,href:null,hrefLang:null,htmlFor:he,httpEquiv:he,id:null,imageSizes:null,imageSrcSet:null,inert:J,inputMode:null,integrity:null,is:null,isMap:J,itemId:null,itemProp:he,itemRef:he,itemScope:J,itemType:he,kind:null,label:null,lang:null,language:null,list:null,loading:null,loop:J,low:j,manifest:null,max:null,maxLength:j,media:null,method:null,min:null,minLength:j,multiple:J,muted:J,name:null,nonce:null,noModule:J,noValidate:J,onAbort:null,onAfterPrint:null,onAuxClick:null,onBeforeMatch:null,onBeforePrint:null,onBeforeToggle:null,onBeforeUnload:null,onBlur:null,onCancel:null,onCanPlay:null,onCanPlayThrough:null,onChange:null,onClick:null,onClose:null,onContextLost:null,onContextMenu:null,onContextRestored:null,onCopy:null,onCueChange:null,onCut:null,onDblClick:null,onDrag:null,onDragEnd:null,onDragEnter:null,onDragExit:null,onDragLeave:null,onDragOver:null,onDragStart:null,onDrop:null,onDurationChange:null,onEmptied:null,onEnded:null,onError:null,onFocus:null,onFormData:null,onHashChange:null,onInput:null,onInvalid:null,onKeyDown:null,onKeyPress:null,onKeyUp:null,onLanguageChange:null,onLoad:null,onLoadedData:null,onLoadedMetadata:null,onLoadEnd:null,onLoadStart:null,onMessage:null,onMessageError:null,onMouseDown:null,onMouseEnter:null,onMouseLeave:null,onMouseMove:null,onMouseOut:null,onMouseOver:null,onMouseUp:null,onOffline:null,onOnline:null,onPageHide:null,onPageShow:null,onPaste:null,onPause:null,onPlay:null,onPlaying:null,onPopState:null,onProgress:null,onRateChange:null,onRejectionHandled:null,onReset:null,onResize:null,onScroll:null,onScrollEnd:null,onSecurityPolicyViolation:null,onSeeked:null,onSeeking:null,onSelect:null,onSlotChange:null,onStalled:null,onStorage:null,onSubmit:null,onSuspend:null,onTimeUpdate:null,onToggle:null,onUnhandledRejection:null,onUnload:null,onVolumeChange:null,onWaiting:null,onWheel:null,open:J,optimum:j,pattern:null,ping:he,placeholder:null,playsInline:J,popover:null,popoverTarget:null,popoverTargetAction:null,poster:null,preload:null,readOnly:J,referrerPolicy:null,rel:he,required:J,reversed:J,rows:j,rowSpan:j,sandbox:he,scope:null,scoped:J,seamless:J,selected:J,shadowRootClonable:J,shadowRootDelegatesFocus:J,shadowRootMode:null,shape:null,size:j,sizes:null,slot:null,span:j,spellCheck:ze,src:null,srcDoc:null,srcLang:null,srcSet:null,start:j,step:null,style:null,tabIndex:j,target:null,title:null,translate:null,type:null,typeMustMatch:J,useMap:null,value:ze,width:j,wrap:null,writingSuggestions:null,align:null,aLink:null,archive:he,axis:null,background:null,bgColor:null,border:j,borderColor:null,bottomMargin:j,cellPadding:null,cellSpacing:null,char:null,charOff:null,classId:null,clear:null,code:null,codeBase:null,codeType:null,color:null,compact:J,declare:J,event:null,face:null,frame:null,frameBorder:null,hSpace:j,leftMargin:j,link:null,longDesc:null,lowSrc:null,marginHeight:j,marginWidth:j,noResize:J,noHref:J,noShade:J,noWrap:J,object:null,profile:null,prompt:null,rev:null,rightMargin:j,rules:null,scheme:null,scrolling:ze,standby:null,summary:null,text:null,topMargin:j,valueType:null,version:null,vAlign:null,vLink:null,vSpace:j,allowTransparency:null,autoCorrect:null,autoSave:null,disablePictureInPicture:J,disableRemotePlayback:J,prefix:null,property:null,results:j,security:null,unselectable:null},space:"html",transform:d0}),Ob=ti({attributes:{accentHeight:"accent-height",alignmentBaseline:"alignment-baseline",arabicForm:"arabic-form",baselineShift:"baseline-shift",capHeight:"cap-height",className:"class",clipPath:"clip-path",clipRule:"clip-rule",colorInterpolation:"color-interpolation",colorInterpolationFilters:"color-interpolation-filters",colorProfile:"color-profile",colorRendering:"color-rendering",crossOrigin:"crossorigin",dataType:"datatype",dominantBaseline:"dominant-baseline",enableBackground:"enable-background",fillOpacity:"fill-opacity",fillRule:"fill-rule",floodColor:"flood-color",floodOpacity:"flood-opacity",fontFamily:"font-family",fontSize:"font-size",fontSizeAdjust:"font-size-adjust",fontStretch:"font-stretch",fontStyle:"font-style",fontVariant:"font-variant",fontWeight:"font-weight",glyphName:"glyph-name",glyphOrientationHorizontal:"glyph-orientation-horizontal",glyphOrientationVertical:"glyph-orientation-vertical",hrefLang:"hreflang",horizAdvX:"horiz-adv-x",horizOriginX:"horiz-origin-x",horizOriginY:"horiz-origin-y",imageRendering:"image-rendering",letterSpacing:"letter-spacing",lightingColor:"lighting-color",markerEnd:"marker-end",markerMid:"marker-mid",markerStart:"marker-start",navDown:"nav-down",navDownLeft:"nav-down-left",navDownRight:"nav-down-right",navLeft:"nav-left",navNext:"nav-next",navPrev:"nav-prev",navRight:"nav-right",navUp:"nav-up",navUpLeft:"nav-up-left",navUpRight:"nav-up-right",onAbort:"onabort",onActivate:"onactivate",onAfterPrint:"onafterprint",onBeforePrint:"onbeforeprint",onBegin:"onbegin",onCancel:"oncancel",onCanPlay:"oncanplay",onCanPlayThrough:"oncanplaythrough",onChange:"onchange",onClick:"onclick",onClose:"onclose",onCopy:"oncopy",onCueChange:"oncuechange",onCut:"oncut",onDblClick:"ondblclick",onDrag:"ondrag",onDragEnd:"ondragend",onDragEnter:"ondragenter",onDragExit:"ondragexit",onDragLeave:"ondragleave",onDragOver:"ondragover",onDragStart:"ondragstart",onDrop:"ondrop",onDurationChange:"ondurationchange",onEmptied:"onemptied",onEnd:"onend",onEnded:"onended",onError:"onerror",onFocus:"onfocus",onFocusIn:"onfocusin",onFocusOut:"onfocusout",onHashChange:"onhashchange",onInput:"oninput",onInvalid:"oninvalid",onKeyDown:"onkeydown",onKeyPress:"onkeypress",onKeyUp:"onkeyup",onLoad:"onload",onLoadedData:"onloadeddata",onLoadedMetadata:"onloadedmetadata",onLoadStart:"onloadstart",onMessage:"onmessage",onMouseDown:"onmousedown",onMouseEnter:"onmouseenter",onMouseLeave:"onmouseleave",onMouseMove:"onmousemove",onMouseOut:"onmouseout",onMouseOver:"onmouseover",onMouseUp:"onmouseup",onMouseWheel:"onmousewheel",onOffline:"onoffline",onOnline:"ononline",onPageHide:"onpagehide",onPageShow:"onpageshow",onPaste:"onpaste",onPause:"onpause",onPlay:"onplay",onPlaying:"onplaying",onPopState:"onpopstate",onProgress:"onprogress",onRateChange:"onratechange",onRepeat:"onrepeat",onReset:"onreset",onResize:"onresize",onScroll:"onscroll",onSeeked:"onseeked",onSeeking:"onseeking",onSelect:"onselect",onShow:"onshow",onStalled:"onstalled",onStorage:"onstorage",onSubmit:"onsubmit",onSuspend:"onsuspend",onTimeUpdate:"ontimeupdate",onToggle:"ontoggle",onUnload:"onunload",onVolumeChange:"onvolumechange",onWaiting:"onwaiting",onZoom:"onzoom",overlinePosition:"overline-position",overlineThickness:"overline-thickness",paintOrder:"paint-order",panose1:"panose-1",pointerEvents:"pointer-events",referrerPolicy:"referrerpolicy",renderingIntent:"rendering-intent",shapeRendering:"shape-rendering",stopColor:"stop-color",stopOpacity:"stop-opacity",strikethroughPosition:"strikethrough-position",strikethroughThickness:"strikethrough-thickness",strokeDashArray:"stroke-dasharray",strokeDashOffset:"stroke-dashoffset",strokeLineCap:"stroke-linecap",strokeLineJoin:"stroke-linejoin",strokeMiterLimit:"stroke-miterlimit",strokeOpacity:"stroke-opacity",strokeWidth:"stroke-width",tabIndex:"tabindex",textAnchor:"text-anchor",textDecoration:"text-decoration",textRendering:"text-rendering",transformOrigin:"transform-origin",typeOf:"typeof",underlinePosition:"underline-position",underlineThickness:"underline-thickness",unicodeBidi:"unicode-bidi",unicodeRange:"unicode-range",unitsPerEm:"units-per-em",vAlphabetic:"v-alphabetic",vHanging:"v-hanging",vIdeographic:"v-ideographic",vMathematical:"v-mathematical",vectorEffect:"vector-effect",vertAdvY:"vert-adv-y",vertOriginX:"vert-origin-x",vertOriginY:"vert-origin-y",wordSpacing:"word-spacing",writingMode:"writing-mode",xHeight:"x-height",playbackOrder:"playbackorder",timelineBegin:"timelinebegin"},properties:{about:ut,accentHeight:j,accumulate:null,additive:null,alignmentBaseline:null,alphabetic:j,amplitude:j,arabicForm:null,ascent:j,attributeName:null,attributeType:null,azimuth:j,bandwidth:null,baselineShift:null,baseFrequency:null,baseProfile:null,bbox:null,begin:null,bias:j,by:null,calcMode:null,capHeight:j,className:he,clip:null,clipPath:null,clipPathUnits:null,clipRule:null,color:null,colorInterpolation:null,colorInterpolationFilters:null,colorProfile:null,colorRendering:null,content:null,contentScriptType:null,contentStyleType:null,crossOrigin:null,cursor:null,cx:null,cy:null,d:null,dataType:null,defaultAction:null,descent:j,diffuseConstant:j,direction:null,display:null,dur:null,divisor:j,dominantBaseline:null,download:J,dx:null,dy:null,edgeMode:null,editable:null,elevation:j,enableBackground:null,end:null,event:null,exponent:j,externalResourcesRequired:null,fill:null,fillOpacity:j,fillRule:null,filter:null,filterRes:null,filterUnits:null,floodColor:null,floodOpacity:null,focusable:null,focusHighlight:null,fontFamily:null,fontSize:null,fontSizeAdjust:null,fontStretch:null,fontStyle:null,fontVariant:null,fontWeight:null,format:null,fr:null,from:null,fx:null,fy:null,g1:Mr,g2:Mr,glyphName:Mr,glyphOrientationHorizontal:null,glyphOrientationVertical:null,glyphRef:null,gradientTransform:null,gradientUnits:null,handler:null,hanging:j,hatchContentUnits:null,hatchUnits:null,height:null,href:null,hrefLang:null,horizAdvX:j,horizOriginX:j,horizOriginY:j,id:null,ideographic:j,imageRendering:null,initialVisibility:null,in:null,in2:null,intercept:j,k:j,k1:j,k2:j,k3:j,k4:j,kernelMatrix:ut,kernelUnitLength:null,keyPoints:null,keySplines:null,keyTimes:null,kerning:null,lang:null,lengthAdjust:null,letterSpacing:null,lightingColor:null,limitingConeAngle:j,local:null,markerEnd:null,markerMid:null,markerStart:null,markerHeight:null,markerUnits:null,markerWidth:null,mask:null,maskContentUnits:null,maskUnits:null,mathematical:null,max:null,media:null,mediaCharacterEncoding:null,mediaContentEncodings:null,mediaSize:j,mediaTime:null,method:null,min:null,mode:null,name:null,navDown:null,navDownLeft:null,navDownRight:null,navLeft:null,navNext:null,navPrev:null,navRight:null,navUp:null,navUpLeft:null,navUpRight:null,numOctaves:null,observer:null,offset:null,onAbort:null,onActivate:null,onAfterPrint:null,onBeforePrint:null,onBegin:null,onCancel:null,onCanPlay:null,onCanPlayThrough:null,onChange:null,onClick:null,onClose:null,onCopy:null,onCueChange:null,onCut:null,onDblClick:null,onDrag:null,onDragEnd:null,onDragEnter:null,onDragExit:null,onDragLeave:null,onDragOver:null,onDragStart:null,onDrop:null,onDurationChange:null,onEmptied:null,onEnd:null,onEnded:null,onError:null,onFocus:null,onFocusIn:null,onFocusOut:null,onHashChange:null,onInput:null,onInvalid:null,onKeyDown:null,onKeyPress:null,onKeyUp:null,onLoad:null,onLoadedData:null,onLoadedMetadata:null,onLoadStart:null,onMessage:null,onMouseDown:null,onMouseEnter:null,onMouseLeave:null,onMouseMove:null,onMouseOut:null,onMouseOver:null,onMouseUp:null,onMouseWheel:null,onOffline:null,onOnline:null,onPageHide:null,onPageShow:null,onPaste:null,onPause:null,onPlay:null,onPlaying:null,onPopState:null,onProgress:null,onRateChange:null,onRepeat:null,onReset:null,onResize:null,onScroll:null,onSeeked:null,onSeeking:null,onSelect:null,onShow:null,onStalled:null,onStorage:null,onSubmit:null,onSuspend:null,onTimeUpdate:null,onToggle:null,onUnload:null,onVolumeChange:null,onWaiting:null,onZoom:null,opacity:null,operator:null,order:null,orient:null,orientation:null,origin:null,overflow:null,overlay:null,overlinePosition:j,overlineThickness:j,paintOrder:null,panose1:null,path:null,pathLength:j,patternContentUnits:null,patternTransform:null,patternUnits:null,phase:null,ping:he,pitch:null,playbackOrder:null,pointerEvents:null,points:null,pointsAtX:j,pointsAtY:j,pointsAtZ:j,preserveAlpha:null,preserveAspectRatio:null,primitiveUnits:null,propagate:null,property:ut,r:null,radius:null,referrerPolicy:null,refX:null,refY:null,rel:ut,rev:ut,renderingIntent:null,repeatCount:null,repeatDur:null,requiredExtensions:ut,requiredFeatures:ut,requiredFonts:ut,requiredFormats:ut,resource:null,restart:null,result:null,rotate:null,rx:null,ry:null,scale:null,seed:null,shapeRendering:null,side:null,slope:null,snapshotTime:null,specularConstant:j,specularExponent:j,spreadMethod:null,spacing:null,startOffset:null,stdDeviation:null,stemh:null,stemv:null,stitchTiles:null,stopColor:null,stopOpacity:null,strikethroughPosition:j,strikethroughThickness:j,string:null,stroke:null,strokeDashArray:ut,strokeDashOffset:null,strokeLineCap:null,strokeLineJoin:null,strokeMiterLimit:j,strokeOpacity:j,strokeWidth:null,style:null,surfaceScale:j,syncBehavior:null,syncBehaviorDefault:null,syncMaster:null,syncTolerance:null,syncToleranceDefault:null,systemLanguage:ut,tabIndex:j,tableValues:null,target:null,targetX:j,targetY:j,textAnchor:null,textDecoration:null,textRendering:null,textLength:null,timelineBegin:null,title:null,transformBehavior:null,type:null,typeOf:ut,to:null,transform:null,transformOrigin:null,u1:null,u2:null,underlinePosition:j,underlineThickness:j,unicode:null,unicodeBidi:null,unicodeRange:null,unitsPerEm:j,values:null,vAlphabetic:j,vMathematical:j,vectorEffect:null,vHanging:j,vIdeographic:j,version:null,vertAdvY:j,vertOriginX:j,vertOriginY:j,viewBox:null,viewTarget:null,visibility:null,width:null,widths:null,wordSpacing:null,writingMode:null,x:null,x1:null,x2:null,xChannelSelector:null,xHeight:j,y:null,y1:null,y2:null,yChannelSelector:null,z:null,zoomAndPan:null},space:"svg",transform:u0}),c0=ti({properties:{xLinkActuate:null,xLinkArcRole:null,xLinkHref:null,xLinkRole:null,xLinkShow:null,xLinkTitle:null,xLinkType:null},space:"xlink",transform(e,t){return"xlink:"+t.slice(5).toLowerCase()}}),p0=ti({attributes:{xmlnsxlink:"xmlns:xlink"},properties:{xmlnsXLink:null,xmlns:null},space:"xmlns",transform:d0}),h0=ti({properties:{xmlBase:null,xmlLang:null,xmlSpace:null},space:"xml",transform(e,t){return"xml:"+t.slice(3).toLowerCase()}}),Wb={classId:"classID",dataType:"datatype",itemId:"itemID",strokeDashArray:"strokeDasharray",strokeDashOffset:"strokeDashoffset",strokeLineCap:"strokeLinecap",strokeLineJoin:"strokeLinejoin",strokeMiterLimit:"strokeMiterlimit",typeOf:"typeof",xLinkActuate:"xlinkActuate",xLinkArcRole:"xlinkArcrole",xLinkHref:"xlinkHref",xLinkRole:"xlinkRole",xLinkShow:"xlinkShow",xLinkTitle:"xlinkTitle",xLinkType:"xlinkType",xmlnsXLink:"xmlnsXlink"},Vb=/[A-Z]/g,bp=/-[a-z]/g,Bb=/^data[-\w.:]+$/i;function Ub(e,t){const n=hu(t);let r=t,i=lt;if(n in e.normal)return e.property[e.normal[n]];if(n.length>4&&n.slice(0,4)==="data"&&Bb.test(t)){if(t.charAt(4)==="-"){const o=t.slice(5).replace(bp,Qb);r="data"+o.charAt(0).toUpperCase()+o.slice(1)}else{const o=t.slice(4);if(!bp.test(o)){let a=o.replace(Vb,Hb);a.charAt(0)!=="-"&&(a="-"+a),t="data"+a}}i=Cd}return new i(r,t)}function Hb(e){return"-"+e.toLowerCase()}function Qb(e){return e.charAt(1).toUpperCase()}const Xb=s0([l0,qb,c0,p0,h0],"html"),Sd=s0([l0,Ob,c0,p0,h0],"svg");function Kb(e){return e.join(" ").trim()}var zd={},vp=/\/\*[^*]*\*+([^/*][^*]*\*+)*\//g,Yb=/\n/g,$b=/^\s*/,Gb=/^(\*?[-#/*\\\w]+(\[[0-9a-z_-]+\])?)\s*/,Zb=/^:\s*/,Jb=/^((?:'(?:\\'|.)*?'|"(?:\\"|.)*?"|\([^)]*?\)|[^};])+)/,ev=/^[;\s]*/,tv=/^\s+|\s+$/g,nv=`
`,kp="/",wp="*",Xn="",rv="comment",iv="declaration";function ov(e,t){if(typeof e!="string")throw new TypeError("First argument must be a string");if(!e)return[];t=t||{};var n=1,r=1;function i(f){var x=f.match(Yb);x&&(n+=x.length);var S=f.lastIndexOf(nv);r=~S?f.length-S:r+f.length}function o(){var f={line:n,column:r};return function(x){return x.position=new a(f),u(),x}}function a(f){this.start=f,this.end={line:n,column:r},this.source=t.source}a.prototype.content=e;function s(f){var x=new Error(t.source+":"+n+":"+r+": "+f);if(x.reason=f,x.filename=t.source,x.line=n,x.column=r,x.source=e,!t.silent)throw x}function l(f){var x=f.exec(e);if(x){var S=x[0];return i(S),e=e.slice(S.length),x}}function u(){l($b)}function c(f){var x;for(f=f||[];x=d();)x!==!1&&f.push(x);return f}function d(){var f=o();if(!(kp!=e.charAt(0)||wp!=e.charAt(1))){for(var x=2;Xn!=e.charAt(x)&&(wp!=e.charAt(x)||kp!=e.charAt(x+1));)++x;if(x+=2,Xn===e.charAt(x-1))return s("End of comment missing");var S=e.slice(2,x-2);return r+=2,i(S),e=e.slice(x),r+=2,f({type:rv,comment:S})}}function h(){var f=o(),x=l(Gb);if(x){if(d(),!l(Zb))return s("property missing ':'");var S=l(Jb),g=f({type:iv,property:Tp(x[0].replace(vp,Xn)),value:S?Tp(S[0].replace(vp,Xn)):Xn});return l(ev),g}}function m(){var f=[];c(f);for(var x;x=h();)x!==!1&&(f.push(x),c(f));return f}return u(),m()}function Tp(e){return e?e.replace(tv,Xn):Xn}var av=ov,sv=la&&la.__importDefault||function(e){return e&&e.__esModule?e:{default:e}};Object.defineProperty(zd,"__esModule",{value:!0});zd.default=uv;const lv=sv(av);function uv(e,t){let n=null;if(!e||typeof e!="string")return n;const r=(0,lv.default)(e),i=typeof t=="function";return r.forEach(o=>{if(o.type!=="declaration")return;const{property:a,value:s}=o;i?t(a,s,o):s&&(n=n||{},n[a]=s)}),n}var as={};Object.defineProperty(as,"__esModule",{value:!0});as.camelCase=void 0;var dv=/^--[a-zA-Z0-9_-]+$/,cv=/-([a-z])/g,pv=/^[^-]+$/,hv=/^-(webkit|moz|ms|o|khtml)-/,mv=/^-(ms)-/,fv=function(e){return!e||pv.test(e)||dv.test(e)},gv=function(e,t){return t.toUpperCase()},Cp=function(e,t){return"".concat(t,"-")},_v=function(e,t){return t===void 0&&(t={}),fv(e)?e:(e=e.toLowerCase(),t.reactCompat?e=e.replace(mv,Cp):e=e.replace(hv,Cp),e.replace(cv,gv))};as.camelCase=_v;var yv=la&&la.__importDefault||function(e){return e&&e.__esModule?e:{default:e}},xv=yv(zd),bv=as;function gu(e,t){var n={};return!e||typeof e!="string"||(0,xv.default)(e,function(r,i){r&&i&&(n[(0,bv.camelCase)(r,t)]=i)}),n}gu.default=gu;var vv=gu;const kv=ar(vv),m0=f0("end"),Ed=f0("start");function f0(e){return t;function t(n){const r=n&&n.position&&n.position[e]||{};if(typeof r.line=="number"&&r.line>0&&typeof r.column=="number"&&r.column>0)return{line:r.line,column:r.column,offset:typeof r.offset=="number"&&r.offset>-1?r.offset:void 0}}}function wv(e){const t=Ed(e),n=m0(e);if(t&&n)return{start:t,end:n}}function Ni(e){return!e||typeof e!="object"?"":"position"in e||"type"in e?Sp(e.position):"start"in e||"end"in e?Sp(e):"line"in e||"column"in e?_u(e):""}function _u(e){return zp(e&&e.line)+":"+zp(e&&e.column)}function Sp(e){return _u(e&&e.start)+"-"+_u(e&&e.end)}function zp(e){return e&&typeof e=="number"?e:1}class Qe extends Error{constructor(t,n,r){super(),typeof n=="string"&&(r=n,n=void 0);let i="",o={},a=!1;if(n&&("line"in n&&"column"in n?o={place:n}:"start"in n&&"end"in n?o={place:n}:"type"in n?o={ancestors:[n],place:n.position}:o={...n}),typeof t=="string"?i=t:!o.cause&&t&&(a=!0,i=t.message,o.cause=t),!o.ruleId&&!o.source&&typeof r=="string"){const l=r.indexOf(":");l===-1?o.ruleId=r:(o.source=r.slice(0,l),o.ruleId=r.slice(l+1))}if(!o.place&&o.ancestors&&o.ancestors){const l=o.ancestors[o.ancestors.length-1];l&&(o.place=l.position)}const s=o.place&&"start"in o.place?o.place.start:o.place;this.ancestors=o.ancestors||void 0,this.cause=o.cause||void 0,this.column=s?s.column:void 0,this.fatal=void 0,this.file="",this.message=i,this.line=s?s.line:void 0,this.name=Ni(o.place)||"1:1",this.place=o.place||void 0,this.reason=this.message,this.ruleId=o.ruleId||void 0,this.source=o.source||void 0,this.stack=a&&o.cause&&typeof o.cause.stack=="string"?o.cause.stack:"",this.actual=void 0,this.expected=void 0,this.note=void 0,this.url=void 0}}Qe.prototype.file="";Qe.prototype.name="";Qe.prototype.reason="";Qe.prototype.message="";Qe.prototype.stack="";Qe.prototype.column=void 0;Qe.prototype.line=void 0;Qe.prototype.ancestors=void 0;Qe.prototype.cause=void 0;Qe.prototype.fatal=void 0;Qe.prototype.place=void 0;Qe.prototype.ruleId=void 0;Qe.prototype.source=void 0;const Pd={}.hasOwnProperty,Tv=new Map,Cv=/[A-Z]/g,Sv=new Set(["table","tbody","thead","tfoot","tr"]),zv=new Set(["td","th"]),g0="https://github.com/syntax-tree/hast-util-to-jsx-runtime";function Ev(e,t){if(!t||t.Fragment===void 0)throw new TypeError("Expected `Fragment` in options");const n=t.filePath||void 0;let r;if(t.development){if(typeof t.jsxDEV!="function")throw new TypeError("Expected `jsxDEV` in options when `development: true`");r=Iv(n,t.jsxDEV)}else{if(typeof t.jsx!="function")throw new TypeError("Expected `jsx` in production options");if(typeof t.jsxs!="function")throw new TypeError("Expected `jsxs` in production options");r=Fv(n,t.jsx,t.jsxs)}const i={Fragment:t.Fragment,ancestors:[],components:t.components||{},create:r,elementAttributeNameCase:t.elementAttributeNameCase||"react",evaluater:t.createEvaluater?t.createEvaluater():void 0,filePath:n,ignoreInvalidStyle:t.ignoreInvalidStyle||!1,passKeys:t.passKeys!==!1,passNode:t.passNode||!1,schema:t.space==="svg"?Sd:Xb,stylePropertyNameCase:t.stylePropertyNameCase||"dom",tableCellAlignToStyle:t.tableCellAlignToStyle!==!1},o=_0(i,e,void 0);return o&&typeof o!="string"?o:i.create(e,i.Fragment,{children:o||void 0},void 0)}function _0(e,t,n){if(t.type==="element")return Pv(e,t,n);if(t.type==="mdxFlowExpression"||t.type==="mdxTextExpression")return Nv(e,t);if(t.type==="mdxJsxFlowElement"||t.type==="mdxJsxTextElement")return Lv(e,t,n);if(t.type==="mdxjsEsm")return Av(e,t);if(t.type==="root")return jv(e,t,n);if(t.type==="text")return Rv(e,t)}function Pv(e,t,n){const r=e.schema;let i=r;t.tagName.toLowerCase()==="svg"&&r.space==="html"&&(i=Sd,e.schema=i),e.ancestors.push(t);const o=x0(e,t.tagName,!1),a=Dv(e,t);let s=Ad(e,t);return Sv.has(t.tagName)&&(s=s.filter(function(l){return typeof l=="string"?!Db(l):!0})),y0(e,a,o,t),Nd(a,s),e.ancestors.pop(),e.schema=r,e.create(t,o,a,n)}function Nv(e,t){if(t.data&&t.data.estree&&e.evaluater){const r=t.data.estree.body[0];return r.type,e.evaluater.evaluateExpression(r.expression)}Ji(e,t.position)}function Av(e,t){if(t.data&&t.data.estree&&e.evaluater)return e.evaluater.evaluateProgram(t.data.estree);Ji(e,t.position)}function Lv(e,t,n){const r=e.schema;let i=r;t.name==="svg"&&r.space==="html"&&(i=Sd,e.schema=i),e.ancestors.push(t);const o=t.name===null?e.Fragment:x0(e,t.name,!0),a=Mv(e,t),s=Ad(e,t);return y0(e,a,o,t),Nd(a,s),e.ancestors.pop(),e.schema=r,e.create(t,o,a,n)}function jv(e,t,n){const r={};return Nd(r,Ad(e,t)),e.create(t,e.Fragment,r,n)}function Rv(e,t){return t.value}function y0(e,t,n,r){typeof n!="string"&&n!==e.Fragment&&e.passNode&&(t.node=r)}function Nd(e,t){if(t.length>0){const n=t.length>1?t:t[0];n&&(e.children=n)}}function Fv(e,t,n){return r;function r(i,o,a,s){const u=Array.isArray(a.children)?n:t;return s?u(o,a,s):u(o,a)}}function Iv(e,t){return n;function n(r,i,o,a){const s=Array.isArray(o.children),l=Ed(r);return t(i,o,a,s,{columnNumber:l?l.column-1:void 0,fileName:e,lineNumber:l?l.line:void 0},void 0)}}function Dv(e,t){const n={};let r,i;for(i in t.properties)if(i!=="children"&&Pd.call(t.properties,i)){const o=qv(e,i,t.properties[i]);if(o){const[a,s]=o;e.tableCellAlignToStyle&&a==="align"&&typeof s=="string"&&zv.has(t.tagName)?r=s:n[a]=s}}if(r){const o=n.style||(n.style={});o[e.stylePropertyNameCase==="css"?"text-align":"textAlign"]=r}return n}function Mv(e,t){const n={};for(const r of t.attributes)if(r.type==="mdxJsxExpressionAttribute")if(r.data&&r.data.estree&&e.evaluater){const o=r.data.estree.body[0];o.type;const a=o.expression;a.type;const s=a.properties[0];s.type,Object.assign(n,e.evaluater.evaluateExpression(s.argument))}else Ji(e,t.position);else{const i=r.name;let o;if(r.value&&typeof r.value=="object")if(r.value.data&&r.value.data.estree&&e.evaluater){const s=r.value.data.estree.body[0];s.type,o=e.evaluater.evaluateExpression(s.expression)}else Ji(e,t.position);else o=r.value===null?!0:r.value;n[i]=o}return n}function Ad(e,t){const n=[];let r=-1;const i=e.passKeys?new Map:Tv;for(;++r<t.children.length;){const o=t.children[r];let a;if(e.passKeys){const l=o.type==="element"?o.tagName:o.type==="mdxJsxFlowElement"||o.type==="mdxJsxTextElement"?o.name:void 0;if(l){const u=i.get(l)||0;a=l+"-"+u,i.set(l,u+1)}}const s=_0(e,o,a);s!==void 0&&n.push(s)}return n}function qv(e,t,n){const r=Ub(e.schema,t);if(!(n==null||typeof n=="number"&&Number.isNaN(n))){if(Array.isArray(n)&&(n=r.commaSeparated?Lb(n):Kb(n)),r.property==="style"){let i=typeof n=="object"?n:Ov(e,String(n));return e.stylePropertyNameCase==="css"&&(i=Wv(i)),["style",i]}return[e.elementAttributeNameCase==="react"&&r.space?Wb[r.property]||r.property:r.attribute,n]}}function Ov(e,t){try{return kv(t,{reactCompat:!0})}catch(n){if(e.ignoreInvalidStyle)return{};const r=n,i=new Qe("Cannot parse `style` attribute",{ancestors:e.ancestors,cause:r,ruleId:"style",source:"hast-util-to-jsx-runtime"});throw i.file=e.filePath||void 0,i.url=g0+"#cannot-parse-style-attribute",i}}function x0(e,t,n){let r;if(!n)r={type:"Literal",value:t};else if(t.includes(".")){const i=t.split(".");let o=-1,a;for(;++o<i.length;){const s=_p(i[o])?{type:"Identifier",name:i[o]}:{type:"Literal",value:i[o]};a=a?{type:"MemberExpression",object:a,property:s,computed:!!(o&&s.type==="Literal"),optional:!1}:s}r=a}else r=_p(t)&&!/^[a-z]/.test(t)?{type:"Identifier",name:t}:{type:"Literal",value:t};if(r.type==="Literal"){const i=r.value;return Pd.call(e.components,i)?e.components[i]:i}if(e.evaluater)return e.evaluater.evaluateExpression(r);Ji(e)}function Ji(e,t){const n=new Qe("Cannot handle MDX estrees without `createEvaluater`",{ancestors:e.ancestors,place:t,ruleId:"mdx-estree",source:"hast-util-to-jsx-runtime"});throw n.file=e.filePath||void 0,n.url=g0+"#cannot-handle-mdx-estrees-without-createevaluater",n}function Wv(e){const t={};let n;for(n in e)Pd.call(e,n)&&(t[Vv(n)]=e[n]);return t}function Vv(e){let t=e.replace(Cv,Bv);return t.slice(0,3)==="ms-"&&(t="-"+t),t}function Bv(e){return"-"+e.toLowerCase()}const Qs={action:["form"],cite:["blockquote","del","ins","q"],data:["object"],formAction:["button","input"],href:["a","area","base","link"],icon:["menuitem"],itemId:null,manifest:["html"],ping:["a","area"],poster:["video"],src:["audio","embed","iframe","img","input","script","source","track","video"]},Uv={};function Ld(e,t){const n=Uv,r=typeof n.includeImageAlt=="boolean"?n.includeImageAlt:!0,i=typeof n.includeHtml=="boolean"?n.includeHtml:!0;return b0(e,r,i)}function b0(e,t,n){if(Hv(e)){if("value"in e)return e.type==="html"&&!n?"":e.value;if(t&&"alt"in e&&e.alt)return e.alt;if("children"in e)return Ep(e.children,t,n)}return Array.isArray(e)?Ep(e,t,n):""}function Ep(e,t,n){const r=[];let i=-1;for(;++i<e.length;)r[i]=b0(e[i],t,n);return r.join("")}function Hv(e){return!!(e&&typeof e=="object")}const Pp=document.createElement("i");function jd(e){const t="&"+e+";";Pp.innerHTML=t;const n=Pp.textContent;return n.charCodeAt(n.length-1)===59&&e!=="semi"||n===t?!1:n}function mt(e,t,n,r){const i=e.length;let o=0,a;if(t<0?t=-t>i?0:i+t:t=t>i?i:t,n=n>0?n:0,r.length<1e4)a=Array.from(r),a.unshift(t,n),e.splice(...a);else for(n&&e.splice(t,n);o<r.length;)a=r.slice(o,o+1e4),a.unshift(t,0),e.splice(...a),o+=1e4,t+=1e4}function wt(e,t){return e.length>0?(mt(e,e.length,0,t),e):t}const Np={}.hasOwnProperty;function v0(e){const t={};let n=-1;for(;++n<e.length;)Qv(t,e[n]);return t}function Qv(e,t){let n;for(n in t){const i=(Np.call(e,n)?e[n]:void 0)||(e[n]={}),o=t[n];let a;if(o)for(a in o){Np.call(i,a)||(i[a]=[]);const s=o[a];Xv(i[a],Array.isArray(s)?s:s?[s]:[])}}}function Xv(e,t){let n=-1;const r=[];for(;++n<t.length;)(t[n].add==="after"?e:r).push(t[n]);mt(e,0,0,r)}function k0(e,t){const n=Number.parseInt(e,t);return n<9||n===11||n>13&&n<32||n>126&&n<160||n>55295&&n<57344||n>64975&&n<65008||(n&65535)===65535||(n&65535)===65534||n>1114111?"�":String.fromCodePoint(n)}function Ot(e){return e.replace(/[\t\n\r ]+/g," ").replace(/^ | $/g,"").toLowerCase().toUpperCase()}const $e=Wn(/[A-Za-z]/),Ue=Wn(/[\dA-Za-z]/),Kv=Wn(/[#-'*+\--9=?A-Z^-~]/);function Fa(e){return e!==null&&(e<32||e===127)}const yu=Wn(/\d/),Yv=Wn(/[\dA-Fa-f]/),$v=Wn(/[!-/:-@[-`{-~]/);function K(e){return e!==null&&e<-2}function pe(e){return e!==null&&(e<0||e===32)}function ne(e){return e===-2||e===-1||e===32}const ss=Wn(new RegExp("\\p{P}|\\p{S}","u")),or=Wn(/\s/);function Wn(e){return t;function t(n){return n!==null&&n>-1&&e.test(String.fromCharCode(n))}}function ni(e){const t=[];let n=-1,r=0,i=0;for(;++n<e.length;){const o=e.charCodeAt(n);let a="";if(o===37&&Ue(e.charCodeAt(n+1))&&Ue(e.charCodeAt(n+2)))i=2;else if(o<128)/[!#$&-;=?-Z_a-z~]/.test(String.fromCharCode(o))||(a=String.fromCharCode(o));else if(o>55295&&o<57344){const s=e.charCodeAt(n+1);o<56320&&s>56319&&s<57344?(a=String.fromCharCode(o,s),i=1):a="�"}else a=String.fromCharCode(o);a&&(t.push(e.slice(r,n),encodeURIComponent(a)),r=n+i+1,a=""),i&&(n+=i,i=0)}return t.join("")+e.slice(r)}function oe(e,t,n,r){const i=r?r-1:Number.POSITIVE_INFINITY;let o=0;return a;function a(l){return ne(l)?(e.enter(n),s(l)):t(l)}function s(l){return ne(l)&&o++<i?(e.consume(l),s):(e.exit(n),t(l))}}const Gv={tokenize:Zv};function Zv(e){const t=e.attempt(this.parser.constructs.contentInitial,r,i);let n;return t;function r(s){if(s===null){e.consume(s);return}return e.enter("lineEnding"),e.consume(s),e.exit("lineEnding"),oe(e,t,"linePrefix")}function i(s){return e.enter("paragraph"),o(s)}function o(s){const l=e.enter("chunkText",{contentType:"text",previous:n});return n&&(n.next=l),n=l,a(s)}function a(s){if(s===null){e.exit("chunkText"),e.exit("paragraph"),e.consume(s);return}return K(s)?(e.consume(s),e.exit("chunkText"),o):(e.consume(s),a)}}const Jv={tokenize:ek},Ap={tokenize:tk};function ek(e){const t=this,n=[];let r=0,i,o,a;return s;function s(_){if(r<n.length){const w=n[r];return t.containerState=w[1],e.attempt(w[0].continuation,l,u)(_)}return u(_)}function l(_){if(r++,t.containerState._closeFlow){t.containerState._closeFlow=void 0,i&&y();const w=t.events.length;let z=w,k;for(;z--;)if(t.events[z][0]==="exit"&&t.events[z][1].type==="chunkFlow"){k=t.events[z][1].end;break}g(r);let E=w;for(;E<t.events.length;)t.events[E][1].end={...k},E++;return mt(t.events,z+1,0,t.events.slice(w)),t.events.length=E,u(_)}return s(_)}function u(_){if(r===n.length){if(!i)return h(_);if(i.currentConstruct&&i.currentConstruct.concrete)return f(_);t.interrupt=!!(i.currentConstruct&&!i._gfmTableDynamicInterruptHack)}return t.containerState={},e.check(Ap,c,d)(_)}function c(_){return i&&y(),g(r),h(_)}function d(_){return t.parser.lazy[t.now().line]=r!==n.length,a=t.now().offset,f(_)}function h(_){return t.containerState={},e.attempt(Ap,m,f)(_)}function m(_){return r++,n.push([t.currentConstruct,t.containerState]),h(_)}function f(_){if(_===null){i&&y(),g(0),e.consume(_);return}return i=i||t.parser.flow(t.now()),e.enter("chunkFlow",{_tokenizer:i,contentType:"flow",previous:o}),x(_)}function x(_){if(_===null){S(e.exit("chunkFlow"),!0),g(0),e.consume(_);return}return K(_)?(e.consume(_),S(e.exit("chunkFlow")),r=0,t.interrupt=void 0,s):(e.consume(_),x)}function S(_,w){const z=t.sliceStream(_);if(w&&z.push(null),_.previous=o,o&&(o.next=_),o=_,i.defineSkip(_.start),i.write(z),t.parser.lazy[_.start.line]){let k=i.events.length;for(;k--;)if(i.events[k][1].start.offset<a&&(!i.events[k][1].end||i.events[k][1].end.offset>a))return;const E=t.events.length;let N=E,q,T;for(;N--;)if(t.events[N][0]==="exit"&&t.events[N][1].type==="chunkFlow"){if(q){T=t.events[N][1].end;break}q=!0}for(g(r),k=E;k<t.events.length;)t.events[k][1].end={...T},k++;mt(t.events,N+1,0,t.events.slice(E)),t.events.length=k}}function g(_){let w=n.length;for(;w-- >_;){const z=n[w];t.containerState=z[1],z[0].exit.call(t,e)}n.length=_}function y(){i.write([null]),o=void 0,i=void 0,t.containerState._closeFlow=void 0}}function tk(e,t,n){return oe(e,e.attempt(this.parser.constructs.document,t,n),"linePrefix",this.parser.constructs.disable.null.includes("codeIndented")?void 0:4)}function Kr(e){if(e===null||pe(e)||or(e))return 1;if(ss(e))return 2}function ls(e,t,n){const r=[];let i=-1;for(;++i<e.length;){const o=e[i].resolveAll;o&&!r.includes(o)&&(t=o(t,n),r.push(o))}return t}const xu={name:"attention",resolveAll:nk,tokenize:rk};function nk(e,t){let n=-1,r,i,o,a,s,l,u,c;for(;++n<e.length;)if(e[n][0]==="enter"&&e[n][1].type==="attentionSequence"&&e[n][1]._close){for(r=n;r--;)if(e[r][0]==="exit"&&e[r][1].type==="attentionSequence"&&e[r][1]._open&&t.sliceSerialize(e[r][1]).charCodeAt(0)===t.sliceSerialize(e[n][1]).charCodeAt(0)){if((e[r][1]._close||e[n][1]._open)&&(e[n][1].end.offset-e[n][1].start.offset)%3&&!((e[r][1].end.offset-e[r][1].start.offset+e[n][1].end.offset-e[n][1].start.offset)%3))continue;l=e[r][1].end.offset-e[r][1].start.offset>1&&e[n][1].end.offset-e[n][1].start.offset>1?2:1;const d={...e[r][1].end},h={...e[n][1].start};Lp(d,-l),Lp(h,l),a={type:l>1?"strongSequence":"emphasisSequence",start:d,end:{...e[r][1].end}},s={type:l>1?"strongSequence":"emphasisSequence",start:{...e[n][1].start},end:h},o={type:l>1?"strongText":"emphasisText",start:{...e[r][1].end},end:{...e[n][1].start}},i={type:l>1?"strong":"emphasis",start:{...a.start},end:{...s.end}},e[r][1].end={...a.start},e[n][1].start={...s.end},u=[],e[r][1].end.offset-e[r][1].start.offset&&(u=wt(u,[["enter",e[r][1],t],["exit",e[r][1],t]])),u=wt(u,[["enter",i,t],["enter",a,t],["exit",a,t],["enter",o,t]]),u=wt(u,ls(t.parser.constructs.insideSpan.null,e.slice(r+1,n),t)),u=wt(u,[["exit",o,t],["enter",s,t],["exit",s,t],["exit",i,t]]),e[n][1].end.offset-e[n][1].start.offset?(c=2,u=wt(u,[["enter",e[n][1],t],["exit",e[n][1],t]])):c=0,mt(e,r-1,n-r+3,u),n=r+u.length-c-2;break}}for(n=-1;++n<e.length;)e[n][1].type==="attentionSequence"&&(e[n][1].type="data");return e}function rk(e,t){const n=this.parser.constructs.attentionMarkers.null,r=this.previous,i=Kr(r);let o;return a;function a(l){return o=l,e.enter("attentionSequence"),s(l)}function s(l){if(l===o)return e.consume(l),s;const u=e.exit("attentionSequence"),c=Kr(l),d=!c||c===2&&i||n.includes(l),h=!i||i===2&&c||n.includes(r);return u._open=!!(o===42?d:d&&(i||!h)),u._close=!!(o===42?h:h&&(c||!d)),t(l)}}function Lp(e,t){e.column+=t,e.offset+=t,e._bufferIndex+=t}const ik={name:"autolink",tokenize:ok};function ok(e,t,n){let r=0;return i;function i(m){return e.enter("autolink"),e.enter("autolinkMarker"),e.consume(m),e.exit("autolinkMarker"),e.enter("autolinkProtocol"),o}function o(m){return $e(m)?(e.consume(m),a):m===64?n(m):u(m)}function a(m){return m===43||m===45||m===46||Ue(m)?(r=1,s(m)):u(m)}function s(m){return m===58?(e.consume(m),r=0,l):(m===43||m===45||m===46||Ue(m))&&r++<32?(e.consume(m),s):(r=0,u(m))}function l(m){return m===62?(e.exit("autolinkProtocol"),e.enter("autolinkMarker"),e.consume(m),e.exit("autolinkMarker"),e.exit("autolink"),t):m===null||m===32||m===60||Fa(m)?n(m):(e.consume(m),l)}function u(m){return m===64?(e.consume(m),c):Kv(m)?(e.consume(m),u):n(m)}function c(m){return Ue(m)?d(m):n(m)}function d(m){return m===46?(e.consume(m),r=0,c):m===62?(e.exit("autolinkProtocol").type="autolinkEmail",e.enter("autolinkMarker"),e.consume(m),e.exit("autolinkMarker"),e.exit("autolink"),t):h(m)}function h(m){if((m===45||Ue(m))&&r++<63){const f=m===45?h:d;return e.consume(m),f}return n(m)}}const co={partial:!0,tokenize:ak};function ak(e,t,n){return r;function r(o){return ne(o)?oe(e,i,"linePrefix")(o):i(o)}function i(o){return o===null||K(o)?t(o):n(o)}}const w0={continuation:{tokenize:lk},exit:uk,name:"blockQuote",tokenize:sk};function sk(e,t,n){const r=this;return i;function i(a){if(a===62){const s=r.containerState;return s.open||(e.enter("blockQuote",{_container:!0}),s.open=!0),e.enter("blockQuotePrefix"),e.enter("blockQuoteMarker"),e.consume(a),e.exit("blockQuoteMarker"),o}return n(a)}function o(a){return ne(a)?(e.enter("blockQuotePrefixWhitespace"),e.consume(a),e.exit("blockQuotePrefixWhitespace"),e.exit("blockQuotePrefix"),t):(e.exit("blockQuotePrefix"),t(a))}}function lk(e,t,n){const r=this;return i;function i(a){return ne(a)?oe(e,o,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(a):o(a)}function o(a){return e.attempt(w0,t,n)(a)}}function uk(e){e.exit("blockQuote")}const T0={name:"characterEscape",tokenize:dk};function dk(e,t,n){return r;function r(o){return e.enter("characterEscape"),e.enter("escapeMarker"),e.consume(o),e.exit("escapeMarker"),i}function i(o){return $v(o)?(e.enter("characterEscapeValue"),e.consume(o),e.exit("characterEscapeValue"),e.exit("characterEscape"),t):n(o)}}const C0={name:"characterReference",tokenize:ck};function ck(e,t,n){const r=this;let i=0,o,a;return s;function s(d){return e.enter("characterReference"),e.enter("characterReferenceMarker"),e.consume(d),e.exit("characterReferenceMarker"),l}function l(d){return d===35?(e.enter("characterReferenceMarkerNumeric"),e.consume(d),e.exit("characterReferenceMarkerNumeric"),u):(e.enter("characterReferenceValue"),o=31,a=Ue,c(d))}function u(d){return d===88||d===120?(e.enter("characterReferenceMarkerHexadecimal"),e.consume(d),e.exit("characterReferenceMarkerHexadecimal"),e.enter("characterReferenceValue"),o=6,a=Yv,c):(e.enter("characterReferenceValue"),o=7,a=yu,c(d))}function c(d){if(d===59&&i){const h=e.exit("characterReferenceValue");return a===Ue&&!jd(r.sliceSerialize(h))?n(d):(e.enter("characterReferenceMarker"),e.consume(d),e.exit("characterReferenceMarker"),e.exit("characterReference"),t)}return a(d)&&i++<o?(e.consume(d),c):n(d)}}const jp={partial:!0,tokenize:hk},Rp={concrete:!0,name:"codeFenced",tokenize:pk};function pk(e,t,n){const r=this,i={partial:!0,tokenize:z};let o=0,a=0,s;return l;function l(k){return u(k)}function u(k){const E=r.events[r.events.length-1];return o=E&&E[1].type==="linePrefix"?E[2].sliceSerialize(E[1],!0).length:0,s=k,e.enter("codeFenced"),e.enter("codeFencedFence"),e.enter("codeFencedFenceSequence"),c(k)}function c(k){return k===s?(a++,e.consume(k),c):a<3?n(k):(e.exit("codeFencedFenceSequence"),ne(k)?oe(e,d,"whitespace")(k):d(k))}function d(k){return k===null||K(k)?(e.exit("codeFencedFence"),r.interrupt?t(k):e.check(jp,x,w)(k)):(e.enter("codeFencedFenceInfo"),e.enter("chunkString",{contentType:"string"}),h(k))}function h(k){return k===null||K(k)?(e.exit("chunkString"),e.exit("codeFencedFenceInfo"),d(k)):ne(k)?(e.exit("chunkString"),e.exit("codeFencedFenceInfo"),oe(e,m,"whitespace")(k)):k===96&&k===s?n(k):(e.consume(k),h)}function m(k){return k===null||K(k)?d(k):(e.enter("codeFencedFenceMeta"),e.enter("chunkString",{contentType:"string"}),f(k))}function f(k){return k===null||K(k)?(e.exit("chunkString"),e.exit("codeFencedFenceMeta"),d(k)):k===96&&k===s?n(k):(e.consume(k),f)}function x(k){return e.attempt(i,w,S)(k)}function S(k){return e.enter("lineEnding"),e.consume(k),e.exit("lineEnding"),g}function g(k){return o>0&&ne(k)?oe(e,y,"linePrefix",o+1)(k):y(k)}function y(k){return k===null||K(k)?e.check(jp,x,w)(k):(e.enter("codeFlowValue"),_(k))}function _(k){return k===null||K(k)?(e.exit("codeFlowValue"),y(k)):(e.consume(k),_)}function w(k){return e.exit("codeFenced"),t(k)}function z(k,E,N){let q=0;return T;function T(F){return k.enter("lineEnding"),k.consume(F),k.exit("lineEnding"),D}function D(F){return k.enter("codeFencedFence"),ne(F)?oe(k,W,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(F):W(F)}function W(F){return F===s?(k.enter("codeFencedFenceSequence"),$(F)):N(F)}function $(F){return F===s?(q++,k.consume(F),$):q>=a?(k.exit("codeFencedFenceSequence"),ne(F)?oe(k,Z,"whitespace")(F):Z(F)):N(F)}function Z(F){return F===null||K(F)?(k.exit("codeFencedFence"),E(F)):N(F)}}}function hk(e,t,n){const r=this;return i;function i(a){return a===null?n(a):(e.enter("lineEnding"),e.consume(a),e.exit("lineEnding"),o)}function o(a){return r.parser.lazy[r.now().line]?n(a):t(a)}}const Xs={name:"codeIndented",tokenize:fk},mk={partial:!0,tokenize:gk};function fk(e,t,n){const r=this;return i;function i(u){return e.enter("codeIndented"),oe(e,o,"linePrefix",5)(u)}function o(u){const c=r.events[r.events.length-1];return c&&c[1].type==="linePrefix"&&c[2].sliceSerialize(c[1],!0).length>=4?a(u):n(u)}function a(u){return u===null?l(u):K(u)?e.attempt(mk,a,l)(u):(e.enter("codeFlowValue"),s(u))}function s(u){return u===null||K(u)?(e.exit("codeFlowValue"),a(u)):(e.consume(u),s)}function l(u){return e.exit("codeIndented"),t(u)}}function gk(e,t,n){const r=this;return i;function i(a){return r.parser.lazy[r.now().line]?n(a):K(a)?(e.enter("lineEnding"),e.consume(a),e.exit("lineEnding"),i):oe(e,o,"linePrefix",5)(a)}function o(a){const s=r.events[r.events.length-1];return s&&s[1].type==="linePrefix"&&s[2].sliceSerialize(s[1],!0).length>=4?t(a):K(a)?i(a):n(a)}}const _k={name:"codeText",previous:xk,resolve:yk,tokenize:bk};function yk(e){let t=e.length-4,n=3,r,i;if((e[n][1].type==="lineEnding"||e[n][1].type==="space")&&(e[t][1].type==="lineEnding"||e[t][1].type==="space")){for(r=n;++r<t;)if(e[r][1].type==="codeTextData"){e[n][1].type="codeTextPadding",e[t][1].type="codeTextPadding",n+=2,t-=2;break}}for(r=n-1,t++;++r<=t;)i===void 0?r!==t&&e[r][1].type!=="lineEnding"&&(i=r):(r===t||e[r][1].type==="lineEnding")&&(e[i][1].type="codeTextData",r!==i+2&&(e[i][1].end=e[r-1][1].end,e.splice(i+2,r-i-2),t-=r-i-2,r=i+2),i=void 0);return e}function xk(e){return e!==96||this.events[this.events.length-1][1].type==="characterEscape"}function bk(e,t,n){let r=0,i,o;return a;function a(d){return e.enter("codeText"),e.enter("codeTextSequence"),s(d)}function s(d){return d===96?(e.consume(d),r++,s):(e.exit("codeTextSequence"),l(d))}function l(d){return d===null?n(d):d===32?(e.enter("space"),e.consume(d),e.exit("space"),l):d===96?(o=e.enter("codeTextSequence"),i=0,c(d)):K(d)?(e.enter("lineEnding"),e.consume(d),e.exit("lineEnding"),l):(e.enter("codeTextData"),u(d))}function u(d){return d===null||d===32||d===96||K(d)?(e.exit("codeTextData"),l(d)):(e.consume(d),u)}function c(d){return d===96?(e.consume(d),i++,c):i===r?(e.exit("codeTextSequence"),e.exit("codeText"),t(d)):(o.type="codeTextData",u(d))}}class vk{constructor(t){this.left=t?[...t]:[],this.right=[]}get(t){if(t<0||t>=this.left.length+this.right.length)throw new RangeError("Cannot access index `"+t+"` in a splice buffer of size `"+(this.left.length+this.right.length)+"`");return t<this.left.length?this.left[t]:this.right[this.right.length-t+this.left.length-1]}get length(){return this.left.length+this.right.length}shift(){return this.setCursor(0),this.right.pop()}slice(t,n){const r=n??Number.POSITIVE_INFINITY;return r<this.left.length?this.left.slice(t,r):t>this.left.length?this.right.slice(this.right.length-r+this.left.length,this.right.length-t+this.left.length).reverse():this.left.slice(t).concat(this.right.slice(this.right.length-r+this.left.length).reverse())}splice(t,n,r){const i=n||0;this.setCursor(Math.trunc(t));const o=this.right.splice(this.right.length-i,Number.POSITIVE_INFINITY);return r&&hi(this.left,r),o.reverse()}pop(){return this.setCursor(Number.POSITIVE_INFINITY),this.left.pop()}push(t){this.setCursor(Number.POSITIVE_INFINITY),this.left.push(t)}pushMany(t){this.setCursor(Number.POSITIVE_INFINITY),hi(this.left,t)}unshift(t){this.setCursor(0),this.right.push(t)}unshiftMany(t){this.setCursor(0),hi(this.right,t.reverse())}setCursor(t){if(!(t===this.left.length||t>this.left.length&&this.right.length===0||t<0&&this.left.length===0))if(t<this.left.length){const n=this.left.splice(t,Number.POSITIVE_INFINITY);hi(this.right,n.reverse())}else{const n=this.right.splice(this.left.length+this.right.length-t,Number.POSITIVE_INFINITY);hi(this.left,n.reverse())}}}function hi(e,t){let n=0;if(t.length<1e4)e.push(...t);else for(;n<t.length;)e.push(...t.slice(n,n+1e4)),n+=1e4}function S0(e){const t={};let n=-1,r,i,o,a,s,l,u;const c=new vk(e);for(;++n<c.length;){for(;n in t;)n=t[n];if(r=c.get(n),n&&r[1].type==="chunkFlow"&&c.get(n-1)[1].type==="listItemPrefix"&&(l=r[1]._tokenizer.events,o=0,o<l.length&&l[o][1].type==="lineEndingBlank"&&(o+=2),o<l.length&&l[o][1].type==="content"))for(;++o<l.length&&l[o][1].type!=="content";)l[o][1].type==="chunkText"&&(l[o][1]._isInFirstContentOfListItem=!0,o++);if(r[0]==="enter")r[1].contentType&&(Object.assign(t,kk(c,n)),n=t[n],u=!0);else if(r[1]._container){for(o=n,i=void 0;o--;)if(a=c.get(o),a[1].type==="lineEnding"||a[1].type==="lineEndingBlank")a[0]==="enter"&&(i&&(c.get(i)[1].type="lineEndingBlank"),a[1].type="lineEnding",i=o);else if(!(a[1].type==="linePrefix"||a[1].type==="listItemIndent"))break;i&&(r[1].end={...c.get(i)[1].start},s=c.slice(i,n),s.unshift(r),c.splice(i,n-i+1,s))}}return mt(e,0,Number.POSITIVE_INFINITY,c.slice(0)),!u}function kk(e,t){const n=e.get(t)[1],r=e.get(t)[2];let i=t-1;const o=[];let a=n._tokenizer;a||(a=r.parser[n.contentType](n.start),n._contentTypeTextTrailing&&(a._contentTypeTextTrailing=!0));const s=a.events,l=[],u={};let c,d,h=-1,m=n,f=0,x=0;const S=[x];for(;m;){for(;e.get(++i)[1]!==m;);o.push(i),m._tokenizer||(c=r.sliceStream(m),m.next||c.push(null),d&&a.defineSkip(m.start),m._isInFirstContentOfListItem&&(a._gfmTasklistFirstContentOfListItem=!0),a.write(c),m._isInFirstContentOfListItem&&(a._gfmTasklistFirstContentOfListItem=void 0)),d=m,m=m.next}for(m=n;++h<s.length;)s[h][0]==="exit"&&s[h-1][0]==="enter"&&s[h][1].type===s[h-1][1].type&&s[h][1].start.line!==s[h][1].end.line&&(x=h+1,S.push(x),m._tokenizer=void 0,m.previous=void 0,m=m.next);for(a.events=[],m?(m._tokenizer=void 0,m.previous=void 0):S.pop(),h=S.length;h--;){const g=s.slice(S[h],S[h+1]),y=o.pop();l.push([y,y+g.length-1]),e.splice(y,2,g)}for(l.reverse(),h=-1;++h<l.length;)u[f+l[h][0]]=f+l[h][1],f+=l[h][1]-l[h][0]-1;return u}const wk={resolve:Ck,tokenize:Sk},Tk={partial:!0,tokenize:zk};function Ck(e){return S0(e),e}function Sk(e,t){let n;return r;function r(s){return e.enter("content"),n=e.enter("chunkContent",{contentType:"content"}),i(s)}function i(s){return s===null?o(s):K(s)?e.check(Tk,a,o)(s):(e.consume(s),i)}function o(s){return e.exit("chunkContent"),e.exit("content"),t(s)}function a(s){return e.consume(s),e.exit("chunkContent"),n.next=e.enter("chunkContent",{contentType:"content",previous:n}),n=n.next,i}}function zk(e,t,n){const r=this;return i;function i(a){return e.exit("chunkContent"),e.enter("lineEnding"),e.consume(a),e.exit("lineEnding"),oe(e,o,"linePrefix")}function o(a){if(a===null||K(a))return n(a);const s=r.events[r.events.length-1];return!r.parser.constructs.disable.null.includes("codeIndented")&&s&&s[1].type==="linePrefix"&&s[2].sliceSerialize(s[1],!0).length>=4?t(a):e.interrupt(r.parser.constructs.flow,n,t)(a)}}function z0(e,t,n,r,i,o,a,s,l){const u=l||Number.POSITIVE_INFINITY;let c=0;return d;function d(g){return g===60?(e.enter(r),e.enter(i),e.enter(o),e.consume(g),e.exit(o),h):g===null||g===32||g===41||Fa(g)?n(g):(e.enter(r),e.enter(a),e.enter(s),e.enter("chunkString",{contentType:"string"}),x(g))}function h(g){return g===62?(e.enter(o),e.consume(g),e.exit(o),e.exit(i),e.exit(r),t):(e.enter(s),e.enter("chunkString",{contentType:"string"}),m(g))}function m(g){return g===62?(e.exit("chunkString"),e.exit(s),h(g)):g===null||g===60||K(g)?n(g):(e.consume(g),g===92?f:m)}function f(g){return g===60||g===62||g===92?(e.consume(g),m):m(g)}function x(g){return!c&&(g===null||g===41||pe(g))?(e.exit("chunkString"),e.exit(s),e.exit(a),e.exit(r),t(g)):c<u&&g===40?(e.consume(g),c++,x):g===41?(e.consume(g),c--,x):g===null||g===32||g===40||Fa(g)?n(g):(e.consume(g),g===92?S:x)}function S(g){return g===40||g===41||g===92?(e.consume(g),x):x(g)}}function E0(e,t,n,r,i,o){const a=this;let s=0,l;return u;function u(m){return e.enter(r),e.enter(i),e.consume(m),e.exit(i),e.enter(o),c}function c(m){return s>999||m===null||m===91||m===93&&!l||m===94&&!s&&"_hiddenFootnoteSupport"in a.parser.constructs?n(m):m===93?(e.exit(o),e.enter(i),e.consume(m),e.exit(i),e.exit(r),t):K(m)?(e.enter("lineEnding"),e.consume(m),e.exit("lineEnding"),c):(e.enter("chunkString",{contentType:"string"}),d(m))}function d(m){return m===null||m===91||m===93||K(m)||s++>999?(e.exit("chunkString"),c(m)):(e.consume(m),l||(l=!ne(m)),m===92?h:d)}function h(m){return m===91||m===92||m===93?(e.consume(m),s++,d):d(m)}}function P0(e,t,n,r,i,o){let a;return s;function s(h){return h===34||h===39||h===40?(e.enter(r),e.enter(i),e.consume(h),e.exit(i),a=h===40?41:h,l):n(h)}function l(h){return h===a?(e.enter(i),e.consume(h),e.exit(i),e.exit(r),t):(e.enter(o),u(h))}function u(h){return h===a?(e.exit(o),l(a)):h===null?n(h):K(h)?(e.enter("lineEnding"),e.consume(h),e.exit("lineEnding"),oe(e,u,"linePrefix")):(e.enter("chunkString",{contentType:"string"}),c(h))}function c(h){return h===a||h===null||K(h)?(e.exit("chunkString"),u(h)):(e.consume(h),h===92?d:c)}function d(h){return h===a||h===92?(e.consume(h),c):c(h)}}function Ai(e,t){let n;return r;function r(i){return K(i)?(e.enter("lineEnding"),e.consume(i),e.exit("lineEnding"),n=!0,r):ne(i)?oe(e,r,n?"linePrefix":"lineSuffix")(i):t(i)}}const Ek={name:"definition",tokenize:Nk},Pk={partial:!0,tokenize:Ak};function Nk(e,t,n){const r=this;let i;return o;function o(m){return e.enter("definition"),a(m)}function a(m){return E0.call(r,e,s,n,"definitionLabel","definitionLabelMarker","definitionLabelString")(m)}function s(m){return i=Ot(r.sliceSerialize(r.events[r.events.length-1][1]).slice(1,-1)),m===58?(e.enter("definitionMarker"),e.consume(m),e.exit("definitionMarker"),l):n(m)}function l(m){return pe(m)?Ai(e,u)(m):u(m)}function u(m){return z0(e,c,n,"definitionDestination","definitionDestinationLiteral","definitionDestinationLiteralMarker","definitionDestinationRaw","definitionDestinationString")(m)}function c(m){return e.attempt(Pk,d,d)(m)}function d(m){return ne(m)?oe(e,h,"whitespace")(m):h(m)}function h(m){return m===null||K(m)?(e.exit("definition"),r.parser.defined.push(i),t(m)):n(m)}}function Ak(e,t,n){return r;function r(s){return pe(s)?Ai(e,i)(s):n(s)}function i(s){return P0(e,o,n,"definitionTitle","definitionTitleMarker","definitionTitleString")(s)}function o(s){return ne(s)?oe(e,a,"whitespace")(s):a(s)}function a(s){return s===null||K(s)?t(s):n(s)}}const Lk={name:"hardBreakEscape",tokenize:jk};function jk(e,t,n){return r;function r(o){return e.enter("hardBreakEscape"),e.consume(o),i}function i(o){return K(o)?(e.exit("hardBreakEscape"),t(o)):n(o)}}const Rk={name:"headingAtx",resolve:Fk,tokenize:Ik};function Fk(e,t){let n=e.length-2,r=3,i,o;return e[r][1].type==="whitespace"&&(r+=2),n-2>r&&e[n][1].type==="whitespace"&&(n-=2),e[n][1].type==="atxHeadingSequence"&&(r===n-1||n-4>r&&e[n-2][1].type==="whitespace")&&(n-=r+1===n?2:4),n>r&&(i={type:"atxHeadingText",start:e[r][1].start,end:e[n][1].end},o={type:"chunkText",start:e[r][1].start,end:e[n][1].end,contentType:"text"},mt(e,r,n-r+1,[["enter",i,t],["enter",o,t],["exit",o,t],["exit",i,t]])),e}function Ik(e,t,n){let r=0;return i;function i(c){return e.enter("atxHeading"),o(c)}function o(c){return e.enter("atxHeadingSequence"),a(c)}function a(c){return c===35&&r++<6?(e.consume(c),a):c===null||pe(c)?(e.exit("atxHeadingSequence"),s(c)):n(c)}function s(c){return c===35?(e.enter("atxHeadingSequence"),l(c)):c===null||K(c)?(e.exit("atxHeading"),t(c)):ne(c)?oe(e,s,"whitespace")(c):(e.enter("atxHeadingText"),u(c))}function l(c){return c===35?(e.consume(c),l):(e.exit("atxHeadingSequence"),s(c))}function u(c){return c===null||c===35||pe(c)?(e.exit("atxHeadingText"),s(c)):(e.consume(c),u)}}const Dk=["address","article","aside","base","basefont","blockquote","body","caption","center","col","colgroup","dd","details","dialog","dir","div","dl","dt","fieldset","figcaption","figure","footer","form","frame","frameset","h1","h2","h3","h4","h5","h6","head","header","hr","html","iframe","legend","li","link","main","menu","menuitem","nav","noframes","ol","optgroup","option","p","param","search","section","summary","table","tbody","td","tfoot","th","thead","title","tr","track","ul"],Fp=["pre","script","style","textarea"],Mk={concrete:!0,name:"htmlFlow",resolveTo:Wk,tokenize:Vk},qk={partial:!0,tokenize:Uk},Ok={partial:!0,tokenize:Bk};function Wk(e){let t=e.length;for(;t--&&!(e[t][0]==="enter"&&e[t][1].type==="htmlFlow"););return t>1&&e[t-2][1].type==="linePrefix"&&(e[t][1].start=e[t-2][1].start,e[t+1][1].start=e[t-2][1].start,e.splice(t-2,2)),e}function Vk(e,t,n){const r=this;let i,o,a,s,l;return u;function u(v){return c(v)}function c(v){return e.enter("htmlFlow"),e.enter("htmlFlowData"),e.consume(v),d}function d(v){return v===33?(e.consume(v),h):v===47?(e.consume(v),o=!0,x):v===63?(e.consume(v),i=3,r.interrupt?t:b):$e(v)?(e.consume(v),a=String.fromCharCode(v),S):n(v)}function h(v){return v===45?(e.consume(v),i=2,m):v===91?(e.consume(v),i=5,s=0,f):$e(v)?(e.consume(v),i=4,r.interrupt?t:b):n(v)}function m(v){return v===45?(e.consume(v),r.interrupt?t:b):n(v)}function f(v){const te="CDATA[";return v===te.charCodeAt(s++)?(e.consume(v),s===te.length?r.interrupt?t:W:f):n(v)}function x(v){return $e(v)?(e.consume(v),a=String.fromCharCode(v),S):n(v)}function S(v){if(v===null||v===47||v===62||pe(v)){const te=v===47,ce=a.toLowerCase();return!te&&!o&&Fp.includes(ce)?(i=1,r.interrupt?t(v):W(v)):Dk.includes(a.toLowerCase())?(i=6,te?(e.consume(v),g):r.interrupt?t(v):W(v)):(i=7,r.interrupt&&!r.parser.lazy[r.now().line]?n(v):o?y(v):_(v))}return v===45||Ue(v)?(e.consume(v),a+=String.fromCharCode(v),S):n(v)}function g(v){return v===62?(e.consume(v),r.interrupt?t:W):n(v)}function y(v){return ne(v)?(e.consume(v),y):T(v)}function _(v){return v===47?(e.consume(v),T):v===58||v===95||$e(v)?(e.consume(v),w):ne(v)?(e.consume(v),_):T(v)}function w(v){return v===45||v===46||v===58||v===95||Ue(v)?(e.consume(v),w):z(v)}function z(v){return v===61?(e.consume(v),k):ne(v)?(e.consume(v),z):_(v)}function k(v){return v===null||v===60||v===61||v===62||v===96?n(v):v===34||v===39?(e.consume(v),l=v,E):ne(v)?(e.consume(v),k):N(v)}function E(v){return v===l?(e.consume(v),l=null,q):v===null||K(v)?n(v):(e.consume(v),E)}function N(v){return v===null||v===34||v===39||v===47||v===60||v===61||v===62||v===96||pe(v)?z(v):(e.consume(v),N)}function q(v){return v===47||v===62||ne(v)?_(v):n(v)}function T(v){return v===62?(e.consume(v),D):n(v)}function D(v){return v===null||K(v)?W(v):ne(v)?(e.consume(v),D):n(v)}function W(v){return v===45&&i===2?(e.consume(v),X):v===60&&i===1?(e.consume(v),ie):v===62&&i===4?(e.consume(v),V):v===63&&i===3?(e.consume(v),b):v===93&&i===5?(e.consume(v),U):K(v)&&(i===6||i===7)?(e.exit("htmlFlowData"),e.check(qk,H,$)(v)):v===null||K(v)?(e.exit("htmlFlowData"),$(v)):(e.consume(v),W)}function $(v){return e.check(Ok,Z,H)(v)}function Z(v){return e.enter("lineEnding"),e.consume(v),e.exit("lineEnding"),F}function F(v){return v===null||K(v)?$(v):(e.enter("htmlFlowData"),W(v))}function X(v){return v===45?(e.consume(v),b):W(v)}function ie(v){return v===47?(e.consume(v),a="",R):W(v)}function R(v){if(v===62){const te=a.toLowerCase();return Fp.includes(te)?(e.consume(v),V):W(v)}return $e(v)&&a.length<8?(e.consume(v),a+=String.fromCharCode(v),R):W(v)}function U(v){return v===93?(e.consume(v),b):W(v)}function b(v){return v===62?(e.consume(v),V):v===45&&i===2?(e.consume(v),b):W(v)}function V(v){return v===null||K(v)?(e.exit("htmlFlowData"),H(v)):(e.consume(v),V)}function H(v){return e.exit("htmlFlow"),t(v)}}function Bk(e,t,n){const r=this;return i;function i(a){return K(a)?(e.enter("lineEnding"),e.consume(a),e.exit("lineEnding"),o):n(a)}function o(a){return r.parser.lazy[r.now().line]?n(a):t(a)}}function Uk(e,t,n){return r;function r(i){return e.enter("lineEnding"),e.consume(i),e.exit("lineEnding"),e.attempt(co,t,n)}}const Hk={name:"htmlText",tokenize:Qk};function Qk(e,t,n){const r=this;let i,o,a;return s;function s(b){return e.enter("htmlText"),e.enter("htmlTextData"),e.consume(b),l}function l(b){return b===33?(e.consume(b),u):b===47?(e.consume(b),z):b===63?(e.consume(b),_):$e(b)?(e.consume(b),N):n(b)}function u(b){return b===45?(e.consume(b),c):b===91?(e.consume(b),o=0,f):$e(b)?(e.consume(b),y):n(b)}function c(b){return b===45?(e.consume(b),m):n(b)}function d(b){return b===null?n(b):b===45?(e.consume(b),h):K(b)?(a=d,ie(b)):(e.consume(b),d)}function h(b){return b===45?(e.consume(b),m):d(b)}function m(b){return b===62?X(b):b===45?h(b):d(b)}function f(b){const V="CDATA[";return b===V.charCodeAt(o++)?(e.consume(b),o===V.length?x:f):n(b)}function x(b){return b===null?n(b):b===93?(e.consume(b),S):K(b)?(a=x,ie(b)):(e.consume(b),x)}function S(b){return b===93?(e.consume(b),g):x(b)}function g(b){return b===62?X(b):b===93?(e.consume(b),g):x(b)}function y(b){return b===null||b===62?X(b):K(b)?(a=y,ie(b)):(e.consume(b),y)}function _(b){return b===null?n(b):b===63?(e.consume(b),w):K(b)?(a=_,ie(b)):(e.consume(b),_)}function w(b){return b===62?X(b):_(b)}function z(b){return $e(b)?(e.consume(b),k):n(b)}function k(b){return b===45||Ue(b)?(e.consume(b),k):E(b)}function E(b){return K(b)?(a=E,ie(b)):ne(b)?(e.consume(b),E):X(b)}function N(b){return b===45||Ue(b)?(e.consume(b),N):b===47||b===62||pe(b)?q(b):n(b)}function q(b){return b===47?(e.consume(b),X):b===58||b===95||$e(b)?(e.consume(b),T):K(b)?(a=q,ie(b)):ne(b)?(e.consume(b),q):X(b)}function T(b){return b===45||b===46||b===58||b===95||Ue(b)?(e.consume(b),T):D(b)}function D(b){return b===61?(e.consume(b),W):K(b)?(a=D,ie(b)):ne(b)?(e.consume(b),D):q(b)}function W(b){return b===null||b===60||b===61||b===62||b===96?n(b):b===34||b===39?(e.consume(b),i=b,$):K(b)?(a=W,ie(b)):ne(b)?(e.consume(b),W):(e.consume(b),Z)}function $(b){return b===i?(e.consume(b),i=void 0,F):b===null?n(b):K(b)?(a=$,ie(b)):(e.consume(b),$)}function Z(b){return b===null||b===34||b===39||b===60||b===61||b===96?n(b):b===47||b===62||pe(b)?q(b):(e.consume(b),Z)}function F(b){return b===47||b===62||pe(b)?q(b):n(b)}function X(b){return b===62?(e.consume(b),e.exit("htmlTextData"),e.exit("htmlText"),t):n(b)}function ie(b){return e.exit("htmlTextData"),e.enter("lineEnding"),e.consume(b),e.exit("lineEnding"),R}function R(b){return ne(b)?oe(e,U,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(b):U(b)}function U(b){return e.enter("htmlTextData"),a(b)}}const Rd={name:"labelEnd",resolveAll:$k,resolveTo:Gk,tokenize:Zk},Xk={tokenize:Jk},Kk={tokenize:e2},Yk={tokenize:t2};function $k(e){let t=-1;const n=[];for(;++t<e.length;){const r=e[t][1];if(n.push(e[t]),r.type==="labelImage"||r.type==="labelLink"||r.type==="labelEnd"){const i=r.type==="labelImage"?4:2;r.type="data",t+=i}}return e.length!==n.length&&mt(e,0,e.length,n),e}function Gk(e,t){let n=e.length,r=0,i,o,a,s;for(;n--;)if(i=e[n][1],o){if(i.type==="link"||i.type==="labelLink"&&i._inactive)break;e[n][0]==="enter"&&i.type==="labelLink"&&(i._inactive=!0)}else if(a){if(e[n][0]==="enter"&&(i.type==="labelImage"||i.type==="labelLink")&&!i._balanced&&(o=n,i.type!=="labelLink")){r=2;break}}else i.type==="labelEnd"&&(a=n);const l={type:e[o][1].type==="labelLink"?"link":"image",start:{...e[o][1].start},end:{...e[e.length-1][1].end}},u={type:"label",start:{...e[o][1].start},end:{...e[a][1].end}},c={type:"labelText",start:{...e[o+r+2][1].end},end:{...e[a-2][1].start}};return s=[["enter",l,t],["enter",u,t]],s=wt(s,e.slice(o+1,o+r+3)),s=wt(s,[["enter",c,t]]),s=wt(s,ls(t.parser.constructs.insideSpan.null,e.slice(o+r+4,a-3),t)),s=wt(s,[["exit",c,t],e[a-2],e[a-1],["exit",u,t]]),s=wt(s,e.slice(a+1)),s=wt(s,[["exit",l,t]]),mt(e,o,e.length,s),e}function Zk(e,t,n){const r=this;let i=r.events.length,o,a;for(;i--;)if((r.events[i][1].type==="labelImage"||r.events[i][1].type==="labelLink")&&!r.events[i][1]._balanced){o=r.events[i][1];break}return s;function s(h){return o?o._inactive?d(h):(a=r.parser.defined.includes(Ot(r.sliceSerialize({start:o.end,end:r.now()}))),e.enter("labelEnd"),e.enter("labelMarker"),e.consume(h),e.exit("labelMarker"),e.exit("labelEnd"),l):n(h)}function l(h){return h===40?e.attempt(Xk,c,a?c:d)(h):h===91?e.attempt(Kk,c,a?u:d)(h):a?c(h):d(h)}function u(h){return e.attempt(Yk,c,d)(h)}function c(h){return t(h)}function d(h){return o._balanced=!0,n(h)}}function Jk(e,t,n){return r;function r(d){return e.enter("resource"),e.enter("resourceMarker"),e.consume(d),e.exit("resourceMarker"),i}function i(d){return pe(d)?Ai(e,o)(d):o(d)}function o(d){return d===41?c(d):z0(e,a,s,"resourceDestination","resourceDestinationLiteral","resourceDestinationLiteralMarker","resourceDestinationRaw","resourceDestinationString",32)(d)}function a(d){return pe(d)?Ai(e,l)(d):c(d)}function s(d){return n(d)}function l(d){return d===34||d===39||d===40?P0(e,u,n,"resourceTitle","resourceTitleMarker","resourceTitleString")(d):c(d)}function u(d){return pe(d)?Ai(e,c)(d):c(d)}function c(d){return d===41?(e.enter("resourceMarker"),e.consume(d),e.exit("resourceMarker"),e.exit("resource"),t):n(d)}}function e2(e,t,n){const r=this;return i;function i(s){return E0.call(r,e,o,a,"reference","referenceMarker","referenceString")(s)}function o(s){return r.parser.defined.includes(Ot(r.sliceSerialize(r.events[r.events.length-1][1]).slice(1,-1)))?t(s):n(s)}function a(s){return n(s)}}function t2(e,t,n){return r;function r(o){return e.enter("reference"),e.enter("referenceMarker"),e.consume(o),e.exit("referenceMarker"),i}function i(o){return o===93?(e.enter("referenceMarker"),e.consume(o),e.exit("referenceMarker"),e.exit("reference"),t):n(o)}}const n2={name:"labelStartImage",resolveAll:Rd.resolveAll,tokenize:r2};function r2(e,t,n){const r=this;return i;function i(s){return e.enter("labelImage"),e.enter("labelImageMarker"),e.consume(s),e.exit("labelImageMarker"),o}function o(s){return s===91?(e.enter("labelMarker"),e.consume(s),e.exit("labelMarker"),e.exit("labelImage"),a):n(s)}function a(s){return s===94&&"_hiddenFootnoteSupport"in r.parser.constructs?n(s):t(s)}}const i2={name:"labelStartLink",resolveAll:Rd.resolveAll,tokenize:o2};function o2(e,t,n){const r=this;return i;function i(a){return e.enter("labelLink"),e.enter("labelMarker"),e.consume(a),e.exit("labelMarker"),e.exit("labelLink"),o}function o(a){return a===94&&"_hiddenFootnoteSupport"in r.parser.constructs?n(a):t(a)}}const Ks={name:"lineEnding",tokenize:a2};function a2(e,t){return n;function n(r){return e.enter("lineEnding"),e.consume(r),e.exit("lineEnding"),oe(e,t,"linePrefix")}}const ia={name:"thematicBreak",tokenize:s2};function s2(e,t,n){let r=0,i;return o;function o(u){return e.enter("thematicBreak"),a(u)}function a(u){return i=u,s(u)}function s(u){return u===i?(e.enter("thematicBreakSequence"),l(u)):r>=3&&(u===null||K(u))?(e.exit("thematicBreak"),t(u)):n(u)}function l(u){return u===i?(e.consume(u),r++,l):(e.exit("thematicBreakSequence"),ne(u)?oe(e,s,"whitespace")(u):s(u))}}const nt={continuation:{tokenize:c2},exit:h2,name:"list",tokenize:d2},l2={partial:!0,tokenize:m2},u2={partial:!0,tokenize:p2};function d2(e,t,n){const r=this,i=r.events[r.events.length-1];let o=i&&i[1].type==="linePrefix"?i[2].sliceSerialize(i[1],!0).length:0,a=0;return s;function s(m){const f=r.containerState.type||(m===42||m===43||m===45?"listUnordered":"listOrdered");if(f==="listUnordered"?!r.containerState.marker||m===r.containerState.marker:yu(m)){if(r.containerState.type||(r.containerState.type=f,e.enter(f,{_container:!0})),f==="listUnordered")return e.enter("listItemPrefix"),m===42||m===45?e.check(ia,n,u)(m):u(m);if(!r.interrupt||m===49)return e.enter("listItemPrefix"),e.enter("listItemValue"),l(m)}return n(m)}function l(m){return yu(m)&&++a<10?(e.consume(m),l):(!r.interrupt||a<2)&&(r.containerState.marker?m===r.containerState.marker:m===41||m===46)?(e.exit("listItemValue"),u(m)):n(m)}function u(m){return e.enter("listItemMarker"),e.consume(m),e.exit("listItemMarker"),r.containerState.marker=r.containerState.marker||m,e.check(co,r.interrupt?n:c,e.attempt(l2,h,d))}function c(m){return r.containerState.initialBlankLine=!0,o++,h(m)}function d(m){return ne(m)?(e.enter("listItemPrefixWhitespace"),e.consume(m),e.exit("listItemPrefixWhitespace"),h):n(m)}function h(m){return r.containerState.size=o+r.sliceSerialize(e.exit("listItemPrefix"),!0).length,t(m)}}function c2(e,t,n){const r=this;return r.containerState._closeFlow=void 0,e.check(co,i,o);function i(s){return r.containerState.furtherBlankLines=r.containerState.furtherBlankLines||r.containerState.initialBlankLine,oe(e,t,"listItemIndent",r.containerState.size+1)(s)}function o(s){return r.containerState.furtherBlankLines||!ne(s)?(r.containerState.furtherBlankLines=void 0,r.containerState.initialBlankLine=void 0,a(s)):(r.containerState.furtherBlankLines=void 0,r.containerState.initialBlankLine=void 0,e.attempt(u2,t,a)(s))}function a(s){return r.containerState._closeFlow=!0,r.interrupt=void 0,oe(e,e.attempt(nt,t,n),"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(s)}}function p2(e,t,n){const r=this;return oe(e,i,"listItemIndent",r.containerState.size+1);function i(o){const a=r.events[r.events.length-1];return a&&a[1].type==="listItemIndent"&&a[2].sliceSerialize(a[1],!0).length===r.containerState.size?t(o):n(o)}}function h2(e){e.exit(this.containerState.type)}function m2(e,t,n){const r=this;return oe(e,i,"listItemPrefixWhitespace",r.parser.constructs.disable.null.includes("codeIndented")?void 0:5);function i(o){const a=r.events[r.events.length-1];return!ne(o)&&a&&a[1].type==="listItemPrefixWhitespace"?t(o):n(o)}}const Ip={name:"setextUnderline",resolveTo:f2,tokenize:g2};function f2(e,t){let n=e.length,r,i,o;for(;n--;)if(e[n][0]==="enter"){if(e[n][1].type==="content"){r=n;break}e[n][1].type==="paragraph"&&(i=n)}else e[n][1].type==="content"&&e.splice(n,1),!o&&e[n][1].type==="definition"&&(o=n);const a={type:"setextHeading",start:{...e[r][1].start},end:{...e[e.length-1][1].end}};return e[i][1].type="setextHeadingText",o?(e.splice(i,0,["enter",a,t]),e.splice(o+1,0,["exit",e[r][1],t]),e[r][1].end={...e[o][1].end}):e[r][1]=a,e.push(["exit",a,t]),e}function g2(e,t,n){const r=this;let i;return o;function o(u){let c=r.events.length,d;for(;c--;)if(r.events[c][1].type!=="lineEnding"&&r.events[c][1].type!=="linePrefix"&&r.events[c][1].type!=="content"){d=r.events[c][1].type==="paragraph";break}return!r.parser.lazy[r.now().line]&&(r.interrupt||d)?(e.enter("setextHeadingLine"),i=u,a(u)):n(u)}function a(u){return e.enter("setextHeadingLineSequence"),s(u)}function s(u){return u===i?(e.consume(u),s):(e.exit("setextHeadingLineSequence"),ne(u)?oe(e,l,"lineSuffix")(u):l(u))}function l(u){return u===null||K(u)?(e.exit("setextHeadingLine"),t(u)):n(u)}}const _2={tokenize:y2};function y2(e){const t=this,n=e.attempt(co,r,e.attempt(this.parser.constructs.flowInitial,i,oe(e,e.attempt(this.parser.constructs.flow,i,e.attempt(wk,i)),"linePrefix")));return n;function r(o){if(o===null){e.consume(o);return}return e.enter("lineEndingBlank"),e.consume(o),e.exit("lineEndingBlank"),t.currentConstruct=void 0,n}function i(o){if(o===null){e.consume(o);return}return e.enter("lineEnding"),e.consume(o),e.exit("lineEnding"),t.currentConstruct=void 0,n}}const x2={resolveAll:A0()},b2=N0("string"),v2=N0("text");function N0(e){return{resolveAll:A0(e==="text"?k2:void 0),tokenize:t};function t(n){const r=this,i=this.parser.constructs[e],o=n.attempt(i,a,s);return a;function a(c){return u(c)?o(c):s(c)}function s(c){if(c===null){n.consume(c);return}return n.enter("data"),n.consume(c),l}function l(c){return u(c)?(n.exit("data"),o(c)):(n.consume(c),l)}function u(c){if(c===null)return!0;const d=i[c];let h=-1;if(d)for(;++h<d.length;){const m=d[h];if(!m.previous||m.previous.call(r,r.previous))return!0}return!1}}}function A0(e){return t;function t(n,r){let i=-1,o;for(;++i<=n.length;)o===void 0?n[i]&&n[i][1].type==="data"&&(o=i,i++):(!n[i]||n[i][1].type!=="data")&&(i!==o+2&&(n[o][1].end=n[i-1][1].end,n.splice(o+2,i-o-2),i=o+2),o=void 0);return e?e(n,r):n}}function k2(e,t){let n=0;for(;++n<=e.length;)if((n===e.length||e[n][1].type==="lineEnding")&&e[n-1][1].type==="data"){const r=e[n-1][1],i=t.sliceStream(r);let o=i.length,a=-1,s=0,l;for(;o--;){const u=i[o];if(typeof u=="string"){for(a=u.length;u.charCodeAt(a-1)===32;)s++,a--;if(a)break;a=-1}else if(u===-2)l=!0,s++;else if(u!==-1){o++;break}}if(t._contentTypeTextTrailing&&n===e.length&&(s=0),s){const u={type:n===e.length||l||s<2?"lineSuffix":"hardBreakTrailing",start:{_bufferIndex:o?a:r.start._bufferIndex+a,_index:r.start._index+o,line:r.end.line,column:r.end.column-s,offset:r.end.offset-s},end:{...r.end}};r.end={...u.start},r.start.offset===r.end.offset?Object.assign(r,u):(e.splice(n,0,["enter",u,t],["exit",u,t]),n+=2)}n++}return e}const w2={42:nt,43:nt,45:nt,48:nt,49:nt,50:nt,51:nt,52:nt,53:nt,54:nt,55:nt,56:nt,57:nt,62:w0},T2={91:Ek},C2={[-2]:Xs,[-1]:Xs,32:Xs},S2={35:Rk,42:ia,45:[Ip,ia],60:Mk,61:Ip,95:ia,96:Rp,126:Rp},z2={38:C0,92:T0},E2={[-5]:Ks,[-4]:Ks,[-3]:Ks,33:n2,38:C0,42:xu,60:[ik,Hk],91:i2,92:[Lk,T0],93:Rd,95:xu,96:_k},P2={null:[xu,x2]},N2={null:[42,95]},A2={null:[]},L2=Object.freeze(Object.defineProperty({__proto__:null,attentionMarkers:N2,contentInitial:T2,disable:A2,document:w2,flow:S2,flowInitial:C2,insideSpan:P2,string:z2,text:E2},Symbol.toStringTag,{value:"Module"}));function j2(e,t,n){let r={_bufferIndex:-1,_index:0,line:n&&n.line||1,column:n&&n.column||1,offset:n&&n.offset||0};const i={},o=[];let a=[],s=[];const l={attempt:E(z),check:E(k),consume:y,enter:_,exit:w,interrupt:E(k,{interrupt:!0})},u={code:null,containerState:{},defineSkip:x,events:[],now:f,parser:e,previous:null,sliceSerialize:h,sliceStream:m,write:d};let c=t.tokenize.call(u,l);return t.resolveAll&&o.push(t),u;function d(D){return a=wt(a,D),S(),a[a.length-1]!==null?[]:(N(t,0),u.events=ls(o,u.events,u),u.events)}function h(D,W){return F2(m(D),W)}function m(D){return R2(a,D)}function f(){const{_bufferIndex:D,_index:W,line:$,column:Z,offset:F}=r;return{_bufferIndex:D,_index:W,line:$,column:Z,offset:F}}function x(D){i[D.line]=D.column,T()}function S(){let D;for(;r._index<a.length;){const W=a[r._index];if(typeof W=="string")for(D=r._index,r._bufferIndex<0&&(r._bufferIndex=0);r._index===D&&r._bufferIndex<W.length;)g(W.charCodeAt(r._bufferIndex));else g(W)}}function g(D){c=c(D)}function y(D){K(D)?(r.line++,r.column=1,r.offset+=D===-3?2:1,T()):D!==-1&&(r.column++,r.offset++),r._bufferIndex<0?r._index++:(r._bufferIndex++,r._bufferIndex===a[r._index].length&&(r._bufferIndex=-1,r._index++)),u.previous=D}function _(D,W){const $=W||{};return $.type=D,$.start=f(),u.events.push(["enter",$,u]),s.push($),$}function w(D){const W=s.pop();return W.end=f(),u.events.push(["exit",W,u]),W}function z(D,W){N(D,W.from)}function k(D,W){W.restore()}function E(D,W){return $;function $(Z,F,X){let ie,R,U,b;return Array.isArray(Z)?H(Z):"tokenize"in Z?H([Z]):V(Z);function V(se){return et;function et(O){const A=O!==null&&se[O],I=O!==null&&se.null,Q=[...Array.isArray(A)?A:A?[A]:[],...Array.isArray(I)?I:I?[I]:[]];return H(Q)(O)}}function H(se){return ie=se,R=0,se.length===0?X:v(se[R])}function v(se){return et;function et(O){return b=q(),U=se,se.partial||(u.currentConstruct=se),se.name&&u.parser.constructs.disable.null.includes(se.name)?ce():se.tokenize.call(W?Object.assign(Object.create(u),W):u,l,te,ce)(O)}}function te(se){return D(U,b),F}function ce(se){return b.restore(),++R<ie.length?v(ie[R]):X}}}function N(D,W){D.resolveAll&&!o.includes(D)&&o.push(D),D.resolve&&mt(u.events,W,u.events.length-W,D.resolve(u.events.slice(W),u)),D.resolveTo&&(u.events=D.resolveTo(u.events,u))}function q(){const D=f(),W=u.previous,$=u.currentConstruct,Z=u.events.length,F=Array.from(s);return{from:Z,restore:X};function X(){r=D,u.previous=W,u.currentConstruct=$,u.events.length=Z,s=F,T()}}function T(){r.line in i&&r.column<2&&(r.column=i[r.line],r.offset+=i[r.line]-1)}}function R2(e,t){const n=t.start._index,r=t.start._bufferIndex,i=t.end._index,o=t.end._bufferIndex;let a;if(n===i)a=[e[n].slice(r,o)];else{if(a=e.slice(n,i),r>-1){const s=a[0];typeof s=="string"?a[0]=s.slice(r):a.shift()}o>0&&a.push(e[i].slice(0,o))}return a}function F2(e,t){let n=-1;const r=[];let i;for(;++n<e.length;){const o=e[n];let a;if(typeof o=="string")a=o;else switch(o){case-5:{a="\r";break}case-4:{a=`
`;break}case-3:{a=`\r
`;break}case-2:{a=t?" ":"	";break}case-1:{if(!t&&i)continue;a=" ";break}default:a=String.fromCharCode(o)}i=o===-2,r.push(a)}return r.join("")}function I2(e){const r={constructs:v0([L2,...(e||{}).extensions||[]]),content:i(Gv),defined:[],document:i(Jv),flow:i(_2),lazy:{},string:i(b2),text:i(v2)};return r;function i(o){return a;function a(s){return j2(r,o,s)}}}function D2(e){for(;!S0(e););return e}const Dp=/[\0\t\n\r]/g;function M2(){let e=1,t="",n=!0,r;return i;function i(o,a,s){const l=[];let u,c,d,h,m;for(o=t+(typeof o=="string"?o.toString():new TextDecoder(a||void 0).decode(o)),d=0,t="",n&&(o.charCodeAt(0)===65279&&d++,n=void 0);d<o.length;){if(Dp.lastIndex=d,u=Dp.exec(o),h=u&&u.index!==void 0?u.index:o.length,m=o.charCodeAt(h),!u){t=o.slice(d);break}if(m===10&&d===h&&r)l.push(-3),r=void 0;else switch(r&&(l.push(-5),r=void 0),d<h&&(l.push(o.slice(d,h)),e+=h-d),m){case 0:{l.push(65533),e++;break}case 9:{for(c=Math.ceil(e/4)*4,l.push(-2);e++<c;)l.push(-1);break}case 10:{l.push(-4),e=1;break}default:r=!0,e=1}d=h+1}return s&&(r&&l.push(-5),t&&l.push(t),l.push(null)),l}}const q2=/\\([!-/:-@[-`{-~])|&(#(?:\d{1,7}|x[\da-f]{1,6})|[\da-z]{1,31});/gi;function O2(e){return e.replace(q2,W2)}function W2(e,t,n){if(t)return t;if(n.charCodeAt(0)===35){const i=n.charCodeAt(1),o=i===120||i===88;return k0(n.slice(o?2:1),o?16:10)}return jd(n)||e}const L0={}.hasOwnProperty;function V2(e,t,n){return typeof t!="string"&&(n=t,t=void 0),B2(n)(D2(I2(n).document().write(M2()(e,t,!0))))}function B2(e){const t={transforms:[],canContainEols:["emphasis","fragment","heading","paragraph","strong"],enter:{autolink:o(fo),autolinkProtocol:q,autolinkEmail:q,atxHeading:o(yt),blockQuote:o(I),characterEscape:q,characterReference:q,codeFenced:o(Q),codeFencedFenceInfo:a,codeFencedFenceMeta:a,codeIndented:o(Q,a),codeText:o(xe,a),codeTextData:q,data:q,codeFlowValue:q,definition:o(fe),definitionDestinationString:a,definitionLabelString:a,definitionTitleString:a,emphasis:o(Ae),hardBreakEscape:o($t),hardBreakTrailing:o($t),htmlFlow:o(Et,a),htmlFlowData:q,htmlText:o(Et,a),htmlTextData:q,image:o(mo),label:a,link:o(fo),listItem:o(go),listItemValue:h,listOrdered:o(Vn,d),listUnordered:o(Vn),paragraph:o(cr),reference:v,referenceString:a,resourceDestinationString:a,resourceTitleString:a,setextHeading:o(yt),strong:o(_o),thematicBreak:o(hs)},exit:{atxHeading:l(),atxHeadingSequence:z,autolink:l(),autolinkEmail:A,autolinkProtocol:O,blockQuote:l(),characterEscapeValue:T,characterReferenceMarkerHexadecimal:ce,characterReferenceMarkerNumeric:ce,characterReferenceValue:se,characterReference:et,codeFenced:l(S),codeFencedFence:x,codeFencedFenceInfo:m,codeFencedFenceMeta:f,codeFlowValue:T,codeIndented:l(g),codeText:l(F),codeTextData:T,data:T,definition:l(),definitionDestinationString:w,definitionLabelString:y,definitionTitleString:_,emphasis:l(),hardBreakEscape:l(W),hardBreakTrailing:l(W),htmlFlow:l($),htmlFlowData:T,htmlText:l(Z),htmlTextData:T,image:l(ie),label:U,labelText:R,lineEnding:D,link:l(X),listItem:l(),listOrdered:l(),listUnordered:l(),paragraph:l(),referenceString:te,resourceDestinationString:b,resourceTitleString:V,resource:H,setextHeading:l(N),setextHeadingLineSequence:E,setextHeadingText:k,strong:l(),thematicBreak:l()}};j0(t,(e||{}).mdastExtensions||[]);const n={};return r;function r(P){let M={type:"root",children:[]};const G={stack:[M],tokenStack:[],config:t,enter:s,exit:u,buffer:a,resume:c,data:n},ee=[];let le=-1;for(;++le<P.length;)if(P[le][1].type==="listOrdered"||P[le][1].type==="listUnordered")if(P[le][0]==="enter")ee.push(le);else{const tt=ee.pop();le=i(P,tt,le)}for(le=-1;++le<P.length;){const tt=t[P[le][0]];L0.call(tt,P[le][1].type)&&tt[P[le][1].type].call(Object.assign({sliceSerialize:P[le][2].sliceSerialize},G),P[le][1])}if(G.tokenStack.length>0){const tt=G.tokenStack[G.tokenStack.length-1];(tt[1]||Mp).call(G,void 0,tt[0])}for(M.position={start:gn(P.length>0?P[0][1].start:{line:1,column:1,offset:0}),end:gn(P.length>0?P[P.length-2][1].end:{line:1,column:1,offset:0})},le=-1;++le<t.transforms.length;)M=t.transforms[le](M)||M;return M}function i(P,M,G){let ee=M-1,le=-1,tt=!1,Y,Ce,mn,Vt;for(;++ee<=G;){const Xe=P[ee];switch(Xe[1].type){case"listUnordered":case"listOrdered":case"blockQuote":{Xe[0]==="enter"?le++:le--,Vt=void 0;break}case"lineEndingBlank":{Xe[0]==="enter"&&(Y&&!Vt&&!le&&!mn&&(mn=ee),Vt=void 0);break}case"linePrefix":case"listItemValue":case"listItemMarker":case"listItemPrefix":case"listItemPrefixWhitespace":break;default:Vt=void 0}if(!le&&Xe[0]==="enter"&&Xe[1].type==="listItemPrefix"||le===-1&&Xe[0]==="exit"&&(Xe[1].type==="listUnordered"||Xe[1].type==="listOrdered")){if(Y){let pr=ee;for(Ce=void 0;pr--;){const Gt=P[pr];if(Gt[1].type==="lineEnding"||Gt[1].type==="lineEndingBlank"){if(Gt[0]==="exit")continue;Ce&&(P[Ce][1].type="lineEndingBlank",tt=!0),Gt[1].type="lineEnding",Ce=pr}else if(!(Gt[1].type==="linePrefix"||Gt[1].type==="blockQuotePrefix"||Gt[1].type==="blockQuotePrefixWhitespace"||Gt[1].type==="blockQuoteMarker"||Gt[1].type==="listItemIndent"))break}mn&&(!Ce||mn<Ce)&&(Y._spread=!0),Y.end=Object.assign({},Ce?P[Ce][1].start:Xe[1].end),P.splice(Ce||ee,0,["exit",Y,Xe[2]]),ee++,G++}if(Xe[1].type==="listItemPrefix"){const pr={type:"listItem",_spread:!1,start:Object.assign({},Xe[1].start),end:void 0};Y=pr,P.splice(ee,0,["enter",pr,Xe[2]]),ee++,G++,mn=void 0,Vt=!0}}}return P[M][1]._spread=tt,G}function o(P,M){return G;function G(ee){s.call(this,P(ee),ee),M&&M.call(this,ee)}}function a(){this.stack.push({type:"fragment",children:[]})}function s(P,M,G){this.stack[this.stack.length-1].children.push(P),this.stack.push(P),this.tokenStack.push([M,G||void 0]),P.position={start:gn(M.start),end:void 0}}function l(P){return M;function M(G){P&&P.call(this,G),u.call(this,G)}}function u(P,M){const G=this.stack.pop(),ee=this.tokenStack.pop();if(ee)ee[0].type!==P.type&&(M?M.call(this,P,ee[0]):(ee[1]||Mp).call(this,P,ee[0]));else throw new Error("Cannot close `"+P.type+"` ("+Ni({start:P.start,end:P.end})+"): it’s not open");G.position.end=gn(P.end)}function c(){return Ld(this.stack.pop())}function d(){this.data.expectingFirstListItemValue=!0}function h(P){if(this.data.expectingFirstListItemValue){const M=this.stack[this.stack.length-2];M.start=Number.parseInt(this.sliceSerialize(P),10),this.data.expectingFirstListItemValue=void 0}}function m(){const P=this.resume(),M=this.stack[this.stack.length-1];M.lang=P}function f(){const P=this.resume(),M=this.stack[this.stack.length-1];M.meta=P}function x(){this.data.flowCodeInside||(this.buffer(),this.data.flowCodeInside=!0)}function S(){const P=this.resume(),M=this.stack[this.stack.length-1];M.value=P.replace(/^(\r?\n|\r)|(\r?\n|\r)$/g,""),this.data.flowCodeInside=void 0}function g(){const P=this.resume(),M=this.stack[this.stack.length-1];M.value=P.replace(/(\r?\n|\r)$/g,"")}function y(P){const M=this.resume(),G=this.stack[this.stack.length-1];G.label=M,G.identifier=Ot(this.sliceSerialize(P)).toLowerCase()}function _(){const P=this.resume(),M=this.stack[this.stack.length-1];M.title=P}function w(){const P=this.resume(),M=this.stack[this.stack.length-1];M.url=P}function z(P){const M=this.stack[this.stack.length-1];if(!M.depth){const G=this.sliceSerialize(P).length;M.depth=G}}function k(){this.data.setextHeadingSlurpLineEnding=!0}function E(P){const M=this.stack[this.stack.length-1];M.depth=this.sliceSerialize(P).codePointAt(0)===61?1:2}function N(){this.data.setextHeadingSlurpLineEnding=void 0}function q(P){const G=this.stack[this.stack.length-1].children;let ee=G[G.length-1];(!ee||ee.type!=="text")&&(ee=hn(),ee.position={start:gn(P.start),end:void 0},G.push(ee)),this.stack.push(ee)}function T(P){const M=this.stack.pop();M.value+=this.sliceSerialize(P),M.position.end=gn(P.end)}function D(P){const M=this.stack[this.stack.length-1];if(this.data.atHardBreak){const G=M.children[M.children.length-1];G.position.end=gn(P.end),this.data.atHardBreak=void 0;return}!this.data.setextHeadingSlurpLineEnding&&t.canContainEols.includes(M.type)&&(q.call(this,P),T.call(this,P))}function W(){this.data.atHardBreak=!0}function $(){const P=this.resume(),M=this.stack[this.stack.length-1];M.value=P}function Z(){const P=this.resume(),M=this.stack[this.stack.length-1];M.value=P}function F(){const P=this.resume(),M=this.stack[this.stack.length-1];M.value=P}function X(){const P=this.stack[this.stack.length-1];if(this.data.inReference){const M=this.data.referenceType||"shortcut";P.type+="Reference",P.referenceType=M,delete P.url,delete P.title}else delete P.identifier,delete P.label;this.data.referenceType=void 0}function ie(){const P=this.stack[this.stack.length-1];if(this.data.inReference){const M=this.data.referenceType||"shortcut";P.type+="Reference",P.referenceType=M,delete P.url,delete P.title}else delete P.identifier,delete P.label;this.data.referenceType=void 0}function R(P){const M=this.sliceSerialize(P),G=this.stack[this.stack.length-2];G.label=O2(M),G.identifier=Ot(M).toLowerCase()}function U(){const P=this.stack[this.stack.length-1],M=this.resume(),G=this.stack[this.stack.length-1];if(this.data.inReference=!0,G.type==="link"){const ee=P.children;G.children=ee}else G.alt=M}function b(){const P=this.resume(),M=this.stack[this.stack.length-1];M.url=P}function V(){const P=this.resume(),M=this.stack[this.stack.length-1];M.title=P}function H(){this.data.inReference=void 0}function v(){this.data.referenceType="collapsed"}function te(P){const M=this.resume(),G=this.stack[this.stack.length-1];G.label=M,G.identifier=Ot(this.sliceSerialize(P)).toLowerCase(),this.data.referenceType="full"}function ce(P){this.data.characterReferenceType=P.type}function se(P){const M=this.sliceSerialize(P),G=this.data.characterReferenceType;let ee;G?(ee=k0(M,G==="characterReferenceMarkerNumeric"?10:16),this.data.characterReferenceType=void 0):ee=jd(M);const le=this.stack[this.stack.length-1];le.value+=ee}function et(P){const M=this.stack.pop();M.position.end=gn(P.end)}function O(P){T.call(this,P);const M=this.stack[this.stack.length-1];M.url=this.sliceSerialize(P)}function A(P){T.call(this,P);const M=this.stack[this.stack.length-1];M.url="mailto:"+this.sliceSerialize(P)}function I(){return{type:"blockquote",children:[]}}function Q(){return{type:"code",lang:null,meta:null,value:""}}function xe(){return{type:"inlineCode",value:""}}function fe(){return{type:"definition",identifier:"",label:null,title:null,url:""}}function Ae(){return{type:"emphasis",children:[]}}function yt(){return{type:"heading",depth:0,children:[]}}function $t(){return{type:"break"}}function Et(){return{type:"html",value:""}}function mo(){return{type:"image",title:null,url:"",alt:null}}function fo(){return{type:"link",title:null,url:"",children:[]}}function Vn(P){return{type:"list",ordered:P.type==="listOrdered",start:null,spread:P._spread,children:[]}}function go(P){return{type:"listItem",spread:P._spread,checked:null,children:[]}}function cr(){return{type:"paragraph",children:[]}}function _o(){return{type:"strong",children:[]}}function hn(){return{type:"text",value:""}}function hs(){return{type:"thematicBreak"}}}function gn(e){return{line:e.line,column:e.column,offset:e.offset}}function j0(e,t){let n=-1;for(;++n<t.length;){const r=t[n];Array.isArray(r)?j0(e,r):U2(e,r)}}function U2(e,t){let n;for(n in t)if(L0.call(t,n))switch(n){case"canContainEols":{const r=t[n];r&&e[n].push(...r);break}case"transforms":{const r=t[n];r&&e[n].push(...r);break}case"enter":case"exit":{const r=t[n];r&&Object.assign(e[n],r);break}}}function Mp(e,t){throw e?new Error("Cannot close `"+e.type+"` ("+Ni({start:e.start,end:e.end})+"): a different token (`"+t.type+"`, "+Ni({start:t.start,end:t.end})+") is open"):new Error("Cannot close document, a token (`"+t.type+"`, "+Ni({start:t.start,end:t.end})+") is still open")}function H2(e){const t=this;t.parser=n;function n(r){return V2(r,{...t.data("settings"),...e,extensions:t.data("micromarkExtensions")||[],mdastExtensions:t.data("fromMarkdownExtensions")||[]})}}function Q2(e,t){const n={type:"element",tagName:"blockquote",properties:{},children:e.wrap(e.all(t),!0)};return e.patch(t,n),e.applyData(t,n)}function X2(e,t){const n={type:"element",tagName:"br",properties:{},children:[]};return e.patch(t,n),[e.applyData(t,n),{type:"text",value:`
`}]}function K2(e,t){const n=t.value?t.value+`
`:"",r={},i=t.lang?t.lang.split(/\s+/):[];i.length>0&&(r.className=["language-"+i[0]]);let o={type:"element",tagName:"code",properties:r,children:[{type:"text",value:n}]};return t.meta&&(o.data={meta:t.meta}),e.patch(t,o),o=e.applyData(t,o),o={type:"element",tagName:"pre",properties:{},children:[o]},e.patch(t,o),o}function Y2(e,t){const n={type:"element",tagName:"del",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function $2(e,t){const n={type:"element",tagName:"em",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function G2(e,t){const n=typeof e.options.clobberPrefix=="string"?e.options.clobberPrefix:"user-content-",r=String(t.identifier).toUpperCase(),i=ni(r.toLowerCase()),o=e.footnoteOrder.indexOf(r);let a,s=e.footnoteCounts.get(r);s===void 0?(s=0,e.footnoteOrder.push(r),a=e.footnoteOrder.length):a=o+1,s+=1,e.footnoteCounts.set(r,s);const l={type:"element",tagName:"a",properties:{href:"#"+n+"fn-"+i,id:n+"fnref-"+i+(s>1?"-"+s:""),dataFootnoteRef:!0,ariaDescribedBy:["footnote-label"]},children:[{type:"text",value:String(a)}]};e.patch(t,l);const u={type:"element",tagName:"sup",properties:{},children:[l]};return e.patch(t,u),e.applyData(t,u)}function Z2(e,t){const n={type:"element",tagName:"h"+t.depth,properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function J2(e,t){if(e.options.allowDangerousHtml){const n={type:"raw",value:t.value};return e.patch(t,n),e.applyData(t,n)}}function R0(e,t){const n=t.referenceType;let r="]";if(n==="collapsed"?r+="[]":n==="full"&&(r+="["+(t.label||t.identifier)+"]"),t.type==="imageReference")return[{type:"text",value:"!["+t.alt+r}];const i=e.all(t),o=i[0];o&&o.type==="text"?o.value="["+o.value:i.unshift({type:"text",value:"["});const a=i[i.length-1];return a&&a.type==="text"?a.value+=r:i.push({type:"text",value:r}),i}function ew(e,t){const n=String(t.identifier).toUpperCase(),r=e.definitionById.get(n);if(!r)return R0(e,t);const i={src:ni(r.url||""),alt:t.alt};r.title!==null&&r.title!==void 0&&(i.title=r.title);const o={type:"element",tagName:"img",properties:i,children:[]};return e.patch(t,o),e.applyData(t,o)}function tw(e,t){const n={src:ni(t.url)};t.alt!==null&&t.alt!==void 0&&(n.alt=t.alt),t.title!==null&&t.title!==void 0&&(n.title=t.title);const r={type:"element",tagName:"img",properties:n,children:[]};return e.patch(t,r),e.applyData(t,r)}function nw(e,t){const n={type:"text",value:t.value.replace(/\r?\n|\r/g," ")};e.patch(t,n);const r={type:"element",tagName:"code",properties:{},children:[n]};return e.patch(t,r),e.applyData(t,r)}function rw(e,t){const n=String(t.identifier).toUpperCase(),r=e.definitionById.get(n);if(!r)return R0(e,t);const i={href:ni(r.url||"")};r.title!==null&&r.title!==void 0&&(i.title=r.title);const o={type:"element",tagName:"a",properties:i,children:e.all(t)};return e.patch(t,o),e.applyData(t,o)}function iw(e,t){const n={href:ni(t.url)};t.title!==null&&t.title!==void 0&&(n.title=t.title);const r={type:"element",tagName:"a",properties:n,children:e.all(t)};return e.patch(t,r),e.applyData(t,r)}function ow(e,t,n){const r=e.all(t),i=n?aw(n):F0(t),o={},a=[];if(typeof t.checked=="boolean"){const c=r[0];let d;c&&c.type==="element"&&c.tagName==="p"?d=c:(d={type:"element",tagName:"p",properties:{},children:[]},r.unshift(d)),d.children.length>0&&d.children.unshift({type:"text",value:" "}),d.children.unshift({type:"element",tagName:"input",properties:{type:"checkbox",checked:t.checked,disabled:!0},children:[]}),o.className=["task-list-item"]}let s=-1;for(;++s<r.length;){const c=r[s];(i||s!==0||c.type!=="element"||c.tagName!=="p")&&a.push({type:"text",value:`
`}),c.type==="element"&&c.tagName==="p"&&!i?a.push(...c.children):a.push(c)}const l=r[r.length-1];l&&(i||l.type!=="element"||l.tagName!=="p")&&a.push({type:"text",value:`
`});const u={type:"element",tagName:"li",properties:o,children:a};return e.patch(t,u),e.applyData(t,u)}function aw(e){let t=!1;if(e.type==="list"){t=e.spread||!1;const n=e.children;let r=-1;for(;!t&&++r<n.length;)t=F0(n[r])}return t}function F0(e){const t=e.spread;return t??e.children.length>1}function sw(e,t){const n={},r=e.all(t);let i=-1;for(typeof t.start=="number"&&t.start!==1&&(n.start=t.start);++i<r.length;){const a=r[i];if(a.type==="element"&&a.tagName==="li"&&a.properties&&Array.isArray(a.properties.className)&&a.properties.className.includes("task-list-item")){n.className=["contains-task-list"];break}}const o={type:"element",tagName:t.ordered?"ol":"ul",properties:n,children:e.wrap(r,!0)};return e.patch(t,o),e.applyData(t,o)}function lw(e,t){const n={type:"element",tagName:"p",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function uw(e,t){const n={type:"root",children:e.wrap(e.all(t))};return e.patch(t,n),e.applyData(t,n)}function dw(e,t){const n={type:"element",tagName:"strong",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}function cw(e,t){const n=e.all(t),r=n.shift(),i=[];if(r){const a={type:"element",tagName:"thead",properties:{},children:e.wrap([r],!0)};e.patch(t.children[0],a),i.push(a)}if(n.length>0){const a={type:"element",tagName:"tbody",properties:{},children:e.wrap(n,!0)},s=Ed(t.children[1]),l=m0(t.children[t.children.length-1]);s&&l&&(a.position={start:s,end:l}),i.push(a)}const o={type:"element",tagName:"table",properties:{},children:e.wrap(i,!0)};return e.patch(t,o),e.applyData(t,o)}function pw(e,t,n){const r=n?n.children:void 0,o=(r?r.indexOf(t):1)===0?"th":"td",a=n&&n.type==="table"?n.align:void 0,s=a?a.length:t.children.length;let l=-1;const u=[];for(;++l<s;){const d=t.children[l],h={},m=a?a[l]:void 0;m&&(h.align=m);let f={type:"element",tagName:o,properties:h,children:[]};d&&(f.children=e.all(d),e.patch(d,f),f=e.applyData(d,f)),u.push(f)}const c={type:"element",tagName:"tr",properties:{},children:e.wrap(u,!0)};return e.patch(t,c),e.applyData(t,c)}function hw(e,t){const n={type:"element",tagName:"td",properties:{},children:e.all(t)};return e.patch(t,n),e.applyData(t,n)}const qp=9,Op=32;function mw(e){const t=String(e),n=/\r?\n|\r/g;let r=n.exec(t),i=0;const o=[];for(;r;)o.push(Wp(t.slice(i,r.index),i>0,!0),r[0]),i=r.index+r[0].length,r=n.exec(t);return o.push(Wp(t.slice(i),i>0,!1)),o.join("")}function Wp(e,t,n){let r=0,i=e.length;if(t){let o=e.codePointAt(r);for(;o===qp||o===Op;)r++,o=e.codePointAt(r)}if(n){let o=e.codePointAt(i-1);for(;o===qp||o===Op;)i--,o=e.codePointAt(i-1)}return i>r?e.slice(r,i):""}function fw(e,t){const n={type:"text",value:mw(String(t.value))};return e.patch(t,n),e.applyData(t,n)}function gw(e,t){const n={type:"element",tagName:"hr",properties:{},children:[]};return e.patch(t,n),e.applyData(t,n)}const _w={blockquote:Q2,break:X2,code:K2,delete:Y2,emphasis:$2,footnoteReference:G2,heading:Z2,html:J2,imageReference:ew,image:tw,inlineCode:nw,linkReference:rw,link:iw,listItem:ow,list:sw,paragraph:lw,root:uw,strong:dw,table:cw,tableCell:hw,tableRow:pw,text:fw,thematicBreak:gw,toml:Io,yaml:Io,definition:Io,footnoteDefinition:Io};function Io(){}const I0=-1,us=0,Li=1,Ia=2,Fd=3,Id=4,Dd=5,Md=6,D0=7,M0=8,Vp=typeof self=="object"?self:globalThis,yw=(e,t)=>{const n=(i,o)=>(e.set(o,i),i),r=i=>{if(e.has(i))return e.get(i);const[o,a]=t[i];switch(o){case us:case I0:return n(a,i);case Li:{const s=n([],i);for(const l of a)s.push(r(l));return s}case Ia:{const s=n({},i);for(const[l,u]of a)s[r(l)]=r(u);return s}case Fd:return n(new Date(a),i);case Id:{const{source:s,flags:l}=a;return n(new RegExp(s,l),i)}case Dd:{const s=n(new Map,i);for(const[l,u]of a)s.set(r(l),r(u));return s}case Md:{const s=n(new Set,i);for(const l of a)s.add(r(l));return s}case D0:{const{name:s,message:l}=a;return n(new Vp[s](l),i)}case M0:return n(BigInt(a),i);case"BigInt":return n(Object(BigInt(a)),i);case"ArrayBuffer":return n(new Uint8Array(a).buffer,a);case"DataView":{const{buffer:s}=new Uint8Array(a);return n(new DataView(s),a)}}return n(new Vp[o](a),i)};return r},Bp=e=>yw(new Map,e)(0),fr="",{toString:xw}={},{keys:bw}=Object,mi=e=>{const t=typeof e;if(t!=="object"||!e)return[us,t];const n=xw.call(e).slice(8,-1);switch(n){case"Array":return[Li,fr];case"Object":return[Ia,fr];case"Date":return[Fd,fr];case"RegExp":return[Id,fr];case"Map":return[Dd,fr];case"Set":return[Md,fr];case"DataView":return[Li,n]}return n.includes("Array")?[Li,n]:n.includes("Error")?[D0,n]:[Ia,n]},Do=([e,t])=>e===us&&(t==="function"||t==="symbol"),vw=(e,t,n,r)=>{const i=(a,s)=>{const l=r.push(a)-1;return n.set(s,l),l},o=a=>{if(n.has(a))return n.get(a);let[s,l]=mi(a);switch(s){case us:{let c=a;switch(l){case"bigint":s=M0,c=a.toString();break;case"function":case"symbol":if(e)throw new TypeError("unable to serialize "+l);c=null;break;case"undefined":return i([I0],a)}return i([s,c],a)}case Li:{if(l){let h=a;return l==="DataView"?h=new Uint8Array(a.buffer):l==="ArrayBuffer"&&(h=new Uint8Array(a)),i([l,[...h]],a)}const c=[],d=i([s,c],a);for(const h of a)c.push(o(h));return d}case Ia:{if(l)switch(l){case"BigInt":return i([l,a.toString()],a);case"Boolean":case"Number":case"String":return i([l,a.valueOf()],a)}if(t&&"toJSON"in a)return o(a.toJSON());const c=[],d=i([s,c],a);for(const h of bw(a))(e||!Do(mi(a[h])))&&c.push([o(h),o(a[h])]);return d}case Fd:return i([s,a.toISOString()],a);case Id:{const{source:c,flags:d}=a;return i([s,{source:c,flags:d}],a)}case Dd:{const c=[],d=i([s,c],a);for(const[h,m]of a)(e||!(Do(mi(h))||Do(mi(m))))&&c.push([o(h),o(m)]);return d}case Md:{const c=[],d=i([s,c],a);for(const h of a)(e||!Do(mi(h)))&&c.push(o(h));return d}}const{message:u}=a;return i([s,{name:l,message:u}],a)};return o},Up=(e,{json:t,lossy:n}={})=>{const r=[];return vw(!(t||n),!!t,new Map,r)(e),r},Da=typeof structuredClone=="function"?(e,t)=>t&&("json"in t||"lossy"in t)?Bp(Up(e,t)):structuredClone(e):(e,t)=>Bp(Up(e,t));function kw(e,t){const n=[{type:"text",value:"↩"}];return t>1&&n.push({type:"element",tagName:"sup",properties:{},children:[{type:"text",value:String(t)}]}),n}function ww(e,t){return"Back to reference "+(e+1)+(t>1?"-"+t:"")}function Tw(e){const t=typeof e.options.clobberPrefix=="string"?e.options.clobberPrefix:"user-content-",n=e.options.footnoteBackContent||kw,r=e.options.footnoteBackLabel||ww,i=e.options.footnoteLabel||"Footnotes",o=e.options.footnoteLabelTagName||"h2",a=e.options.footnoteLabelProperties||{className:["sr-only"]},s=[];let l=-1;for(;++l<e.footnoteOrder.length;){const u=e.footnoteById.get(e.footnoteOrder[l]);if(!u)continue;const c=e.all(u),d=String(u.identifier).toUpperCase(),h=ni(d.toLowerCase());let m=0;const f=[],x=e.footnoteCounts.get(d);for(;x!==void 0&&++m<=x;){f.length>0&&f.push({type:"text",value:" "});let y=typeof n=="string"?n:n(l,m);typeof y=="string"&&(y={type:"text",value:y}),f.push({type:"element",tagName:"a",properties:{href:"#"+t+"fnref-"+h+(m>1?"-"+m:""),dataFootnoteBackref:"",ariaLabel:typeof r=="string"?r:r(l,m),className:["data-footnote-backref"]},children:Array.isArray(y)?y:[y]})}const S=c[c.length-1];if(S&&S.type==="element"&&S.tagName==="p"){const y=S.children[S.children.length-1];y&&y.type==="text"?y.value+=" ":S.children.push({type:"text",value:" "}),S.children.push(...f)}else c.push(...f);const g={type:"element",tagName:"li",properties:{id:t+"fn-"+h},children:e.wrap(c,!0)};e.patch(u,g),s.push(g)}if(s.length!==0)return{type:"element",tagName:"section",properties:{dataFootnotes:!0,className:["footnotes"]},children:[{type:"element",tagName:o,properties:{...Da(a),id:"footnote-label"},children:[{type:"text",value:i}]},{type:"text",value:`
`},{type:"element",tagName:"ol",properties:{},children:e.wrap(s,!0)},{type:"text",value:`
`}]}}const ds=function(e){if(e==null)return Ew;if(typeof e=="function")return cs(e);if(typeof e=="object")return Array.isArray(e)?Cw(e):Sw(e);if(typeof e=="string")return zw(e);throw new Error("Expected function, string, or object as test")};function Cw(e){const t=[];let n=-1;for(;++n<e.length;)t[n]=ds(e[n]);return cs(r);function r(...i){let o=-1;for(;++o<t.length;)if(t[o].apply(this,i))return!0;return!1}}function Sw(e){const t=e;return cs(n);function n(r){const i=r;let o;for(o in e)if(i[o]!==t[o])return!1;return!0}}function zw(e){return cs(t);function t(n){return n&&n.type===e}}function cs(e){return t;function t(n,r,i){return!!(Pw(n)&&e.call(this,n,typeof r=="number"?r:void 0,i||void 0))}}function Ew(){return!0}function Pw(e){return e!==null&&typeof e=="object"&&"type"in e}const q0=[],Nw=!0,bu=!1,Aw="skip";function O0(e,t,n,r){let i;typeof t=="function"&&typeof n!="function"?(r=n,n=t):i=t;const o=ds(i),a=r?-1:1;s(e,void 0,[])();function s(l,u,c){const d=l&&typeof l=="object"?l:{};if(typeof d.type=="string"){const m=typeof d.tagName=="string"?d.tagName:typeof d.name=="string"?d.name:void 0;Object.defineProperty(h,"name",{value:"node ("+(l.type+(m?"<"+m+">":""))+")"})}return h;function h(){let m=q0,f,x,S;if((!t||o(l,u,c[c.length-1]||void 0))&&(m=Lw(n(l,c)),m[0]===bu))return m;if("children"in l&&l.children){const g=l;if(g.children&&m[0]!==Aw)for(x=(r?g.children.length:-1)+a,S=c.concat(g);x>-1&&x<g.children.length;){const y=g.children[x];if(f=s(y,x,S)(),f[0]===bu)return f;x=typeof f[1]=="number"?f[1]:x+a}}return m}}}function Lw(e){return Array.isArray(e)?e:typeof e=="number"?[Nw,e]:e==null?q0:[e]}function qd(e,t,n,r){let i,o,a;typeof t=="function"&&typeof n!="function"?(o=void 0,a=t,i=n):(o=t,a=n,i=r),O0(e,o,s,i);function s(l,u){const c=u[u.length-1],d=c?c.children.indexOf(l):void 0;return a(l,d,c)}}const vu={}.hasOwnProperty,jw={};function Rw(e,t){const n=t||jw,r=new Map,i=new Map,o=new Map,a={..._w,...n.handlers},s={all:u,applyData:Iw,definitionById:r,footnoteById:i,footnoteCounts:o,footnoteOrder:[],handlers:a,one:l,options:n,patch:Fw,wrap:Mw};return qd(e,function(c){if(c.type==="definition"||c.type==="footnoteDefinition"){const d=c.type==="definition"?r:i,h=String(c.identifier).toUpperCase();d.has(h)||d.set(h,c)}}),s;function l(c,d){const h=c.type,m=s.handlers[h];if(vu.call(s.handlers,h)&&m)return m(s,c,d);if(s.options.passThrough&&s.options.passThrough.includes(h)){if("children"in c){const{children:x,...S}=c,g=Da(S);return g.children=s.all(c),g}return Da(c)}return(s.options.unknownHandler||Dw)(s,c,d)}function u(c){const d=[];if("children"in c){const h=c.children;let m=-1;for(;++m<h.length;){const f=s.one(h[m],c);if(f){if(m&&h[m-1].type==="break"&&(!Array.isArray(f)&&f.type==="text"&&(f.value=Hp(f.value)),!Array.isArray(f)&&f.type==="element")){const x=f.children[0];x&&x.type==="text"&&(x.value=Hp(x.value))}Array.isArray(f)?d.push(...f):d.push(f)}}}return d}}function Fw(e,t){e.position&&(t.position=wv(e))}function Iw(e,t){let n=t;if(e&&e.data){const r=e.data.hName,i=e.data.hChildren,o=e.data.hProperties;if(typeof r=="string")if(n.type==="element")n.tagName=r;else{const a="children"in n?n.children:[n];n={type:"element",tagName:r,properties:{},children:a}}n.type==="element"&&o&&Object.assign(n.properties,Da(o)),"children"in n&&n.children&&i!==null&&i!==void 0&&(n.children=i)}return n}function Dw(e,t){const n=t.data||{},r="value"in t&&!(vu.call(n,"hProperties")||vu.call(n,"hChildren"))?{type:"text",value:t.value}:{type:"element",tagName:"div",properties:{},children:e.all(t)};return e.patch(t,r),e.applyData(t,r)}function Mw(e,t){const n=[];let r=-1;for(t&&n.push({type:"text",value:`
`});++r<e.length;)r&&n.push({type:"text",value:`
`}),n.push(e[r]);return t&&e.length>0&&n.push({type:"text",value:`
`}),n}function Hp(e){let t=0,n=e.charCodeAt(t);for(;n===9||n===32;)t++,n=e.charCodeAt(t);return e.slice(t)}function Qp(e,t){const n=Rw(e,t),r=n.one(e,void 0),i=Tw(n),o=Array.isArray(r)?{type:"root",children:r}:r||{type:"root",children:[]};return i&&o.children.push({type:"text",value:`
`},i),o}function qw(e,t){return e&&"run"in e?async function(n,r){const i=Qp(n,{file:r,...t});await e.run(i,r)}:function(n,r){return Qp(n,{file:r,...e||t})}}function Xp(e){if(e)throw e}var oa=Object.prototype.hasOwnProperty,W0=Object.prototype.toString,Kp=Object.defineProperty,Yp=Object.getOwnPropertyDescriptor,$p=function(t){return typeof Array.isArray=="function"?Array.isArray(t):W0.call(t)==="[object Array]"},Gp=function(t){if(!t||W0.call(t)!=="[object Object]")return!1;var n=oa.call(t,"constructor"),r=t.constructor&&t.constructor.prototype&&oa.call(t.constructor.prototype,"isPrototypeOf");if(t.constructor&&!n&&!r)return!1;var i;for(i in t);return typeof i>"u"||oa.call(t,i)},Zp=function(t,n){Kp&&n.name==="__proto__"?Kp(t,n.name,{enumerable:!0,configurable:!0,value:n.newValue,writable:!0}):t[n.name]=n.newValue},Jp=function(t,n){if(n==="__proto__")if(oa.call(t,n)){if(Yp)return Yp(t,n).value}else return;return t[n]},Ow=function e(){var t,n,r,i,o,a,s=arguments[0],l=1,u=arguments.length,c=!1;for(typeof s=="boolean"&&(c=s,s=arguments[1]||{},l=2),(s==null||typeof s!="object"&&typeof s!="function")&&(s={});l<u;++l)if(t=arguments[l],t!=null)for(n in t)r=Jp(s,n),i=Jp(t,n),s!==i&&(c&&i&&(Gp(i)||(o=$p(i)))?(o?(o=!1,a=r&&$p(r)?r:[]):a=r&&Gp(r)?r:{},Zp(s,{name:n,newValue:e(c,a,i)})):typeof i<"u"&&Zp(s,{name:n,newValue:i}));return s};const Ys=ar(Ow);function ku(e){if(typeof e!="object"||e===null)return!1;const t=Object.getPrototypeOf(e);return(t===null||t===Object.prototype||Object.getPrototypeOf(t)===null)&&!(Symbol.toStringTag in e)&&!(Symbol.iterator in e)}function Ww(){const e=[],t={run:n,use:r};return t;function n(...i){let o=-1;const a=i.pop();if(typeof a!="function")throw new TypeError("Expected function as last argument, not "+a);s(null,...i);function s(l,...u){const c=e[++o];let d=-1;if(l){a(l);return}for(;++d<i.length;)(u[d]===null||u[d]===void 0)&&(u[d]=i[d]);i=u,c?Vw(c,s)(...u):a(null,...u)}}function r(i){if(typeof i!="function")throw new TypeError("Expected `middelware` to be a function, not "+i);return e.push(i),t}}function Vw(e,t){let n;return r;function r(...a){const s=e.length>a.length;let l;s&&a.push(i);try{l=e.apply(this,a)}catch(u){const c=u;if(s&&n)throw c;return i(c)}s||(l&&l.then&&typeof l.then=="function"?l.then(o,i):l instanceof Error?i(l):o(l))}function i(a,...s){n||(n=!0,t(a,...s))}function o(a){i(null,a)}}const Ht={basename:Bw,dirname:Uw,extname:Hw,join:Qw,sep:"/"};function Bw(e,t){if(t!==void 0&&typeof t!="string")throw new TypeError('"ext" argument must be a string');po(e);let n=0,r=-1,i=e.length,o;if(t===void 0||t.length===0||t.length>e.length){for(;i--;)if(e.codePointAt(i)===47){if(o){n=i+1;break}}else r<0&&(o=!0,r=i+1);return r<0?"":e.slice(n,r)}if(t===e)return"";let a=-1,s=t.length-1;for(;i--;)if(e.codePointAt(i)===47){if(o){n=i+1;break}}else a<0&&(o=!0,a=i+1),s>-1&&(e.codePointAt(i)===t.codePointAt(s--)?s<0&&(r=i):(s=-1,r=a));return n===r?r=a:r<0&&(r=e.length),e.slice(n,r)}function Uw(e){if(po(e),e.length===0)return".";let t=-1,n=e.length,r;for(;--n;)if(e.codePointAt(n)===47){if(r){t=n;break}}else r||(r=!0);return t<0?e.codePointAt(0)===47?"/":".":t===1&&e.codePointAt(0)===47?"//":e.slice(0,t)}function Hw(e){po(e);let t=e.length,n=-1,r=0,i=-1,o=0,a;for(;t--;){const s=e.codePointAt(t);if(s===47){if(a){r=t+1;break}continue}n<0&&(a=!0,n=t+1),s===46?i<0?i=t:o!==1&&(o=1):i>-1&&(o=-1)}return i<0||n<0||o===0||o===1&&i===n-1&&i===r+1?"":e.slice(i,n)}function Qw(...e){let t=-1,n;for(;++t<e.length;)po(e[t]),e[t]&&(n=n===void 0?e[t]:n+"/"+e[t]);return n===void 0?".":Xw(n)}function Xw(e){po(e);const t=e.codePointAt(0)===47;let n=Kw(e,!t);return n.length===0&&!t&&(n="."),n.length>0&&e.codePointAt(e.length-1)===47&&(n+="/"),t?"/"+n:n}function Kw(e,t){let n="",r=0,i=-1,o=0,a=-1,s,l;for(;++a<=e.length;){if(a<e.length)s=e.codePointAt(a);else{if(s===47)break;s=47}if(s===47){if(!(i===a-1||o===1))if(i!==a-1&&o===2){if(n.length<2||r!==2||n.codePointAt(n.length-1)!==46||n.codePointAt(n.length-2)!==46){if(n.length>2){if(l=n.lastIndexOf("/"),l!==n.length-1){l<0?(n="",r=0):(n=n.slice(0,l),r=n.length-1-n.lastIndexOf("/")),i=a,o=0;continue}}else if(n.length>0){n="",r=0,i=a,o=0;continue}}t&&(n=n.length>0?n+"/..":"..",r=2)}else n.length>0?n+="/"+e.slice(i+1,a):n=e.slice(i+1,a),r=a-i-1;i=a,o=0}else s===46&&o>-1?o++:o=-1}return n}function po(e){if(typeof e!="string")throw new TypeError("Path must be a string. Received "+JSON.stringify(e))}const Yw={cwd:$w};function $w(){return"/"}function wu(e){return!!(e!==null&&typeof e=="object"&&"href"in e&&e.href&&"protocol"in e&&e.protocol&&e.auth===void 0)}function Gw(e){if(typeof e=="string")e=new URL(e);else if(!wu(e)){const t=new TypeError('The "path" argument must be of type string or an instance of URL. Received `'+e+"`");throw t.code="ERR_INVALID_ARG_TYPE",t}if(e.protocol!=="file:"){const t=new TypeError("The URL must be of scheme file");throw t.code="ERR_INVALID_URL_SCHEME",t}return Zw(e)}function Zw(e){if(e.hostname!==""){const r=new TypeError('File URL host must be "localhost" or empty on darwin');throw r.code="ERR_INVALID_FILE_URL_HOST",r}const t=e.pathname;let n=-1;for(;++n<t.length;)if(t.codePointAt(n)===37&&t.codePointAt(n+1)===50){const r=t.codePointAt(n+2);if(r===70||r===102){const i=new TypeError("File URL path must not include encoded / characters");throw i.code="ERR_INVALID_FILE_URL_PATH",i}}return decodeURIComponent(t)}const $s=["history","path","basename","stem","extname","dirname"];class V0{constructor(t){let n;t?wu(t)?n={path:t}:typeof t=="string"||Jw(t)?n={value:t}:n=t:n={},this.cwd="cwd"in n?"":Yw.cwd(),this.data={},this.history=[],this.messages=[],this.value,this.map,this.result,this.stored;let r=-1;for(;++r<$s.length;){const o=$s[r];o in n&&n[o]!==void 0&&n[o]!==null&&(this[o]=o==="history"?[...n[o]]:n[o])}let i;for(i in n)$s.includes(i)||(this[i]=n[i])}get basename(){return typeof this.path=="string"?Ht.basename(this.path):void 0}set basename(t){Zs(t,"basename"),Gs(t,"basename"),this.path=Ht.join(this.dirname||"",t)}get dirname(){return typeof this.path=="string"?Ht.dirname(this.path):void 0}set dirname(t){eh(this.basename,"dirname"),this.path=Ht.join(t||"",this.basename)}get extname(){return typeof this.path=="string"?Ht.extname(this.path):void 0}set extname(t){if(Gs(t,"extname"),eh(this.dirname,"extname"),t){if(t.codePointAt(0)!==46)throw new Error("`extname` must start with `.`");if(t.includes(".",1))throw new Error("`extname` cannot contain multiple dots")}this.path=Ht.join(this.dirname,this.stem+(t||""))}get path(){return this.history[this.history.length-1]}set path(t){wu(t)&&(t=Gw(t)),Zs(t,"path"),this.path!==t&&this.history.push(t)}get stem(){return typeof this.path=="string"?Ht.basename(this.path,this.extname):void 0}set stem(t){Zs(t,"stem"),Gs(t,"stem"),this.path=Ht.join(this.dirname||"",t+(this.extname||""))}fail(t,n,r){const i=this.message(t,n,r);throw i.fatal=!0,i}info(t,n,r){const i=this.message(t,n,r);return i.fatal=void 0,i}message(t,n,r){const i=new Qe(t,n,r);return this.path&&(i.name=this.path+":"+i.name,i.file=this.path),i.fatal=!1,this.messages.push(i),i}toString(t){return this.value===void 0?"":typeof this.value=="string"?this.value:new TextDecoder(t||void 0).decode(this.value)}}function Gs(e,t){if(e&&e.includes(Ht.sep))throw new Error("`"+t+"` cannot be a path: did not expect `"+Ht.sep+"`")}function Zs(e,t){if(!e)throw new Error("`"+t+"` cannot be empty")}function eh(e,t){if(!e)throw new Error("Setting `"+t+"` requires `path` to be set too")}function Jw(e){return!!(e&&typeof e=="object"&&"byteLength"in e&&"byteOffset"in e)}const eT=function(e){const r=this.constructor.prototype,i=r[e],o=function(){return i.apply(o,arguments)};return Object.setPrototypeOf(o,r),o},tT={}.hasOwnProperty;class Od extends eT{constructor(){super("copy"),this.Compiler=void 0,this.Parser=void 0,this.attachers=[],this.compiler=void 0,this.freezeIndex=-1,this.frozen=void 0,this.namespace={},this.parser=void 0,this.transformers=Ww()}copy(){const t=new Od;let n=-1;for(;++n<this.attachers.length;){const r=this.attachers[n];t.use(...r)}return t.data(Ys(!0,{},this.namespace)),t}data(t,n){return typeof t=="string"?arguments.length===2?(tl("data",this.frozen),this.namespace[t]=n,this):tT.call(this.namespace,t)&&this.namespace[t]||void 0:t?(tl("data",this.frozen),this.namespace=t,this):this.namespace}freeze(){if(this.frozen)return this;const t=this;for(;++this.freezeIndex<this.attachers.length;){const[n,...r]=this.attachers[this.freezeIndex];if(r[0]===!1)continue;r[0]===!0&&(r[0]=void 0);const i=n.call(t,...r);typeof i=="function"&&this.transformers.use(i)}return this.frozen=!0,this.freezeIndex=Number.POSITIVE_INFINITY,this}parse(t){this.freeze();const n=Mo(t),r=this.parser||this.Parser;return Js("parse",r),r(String(n),n)}process(t,n){const r=this;return this.freeze(),Js("process",this.parser||this.Parser),el("process",this.compiler||this.Compiler),n?i(void 0,n):new Promise(i);function i(o,a){const s=Mo(t),l=r.parse(s);r.run(l,s,function(c,d,h){if(c||!d||!h)return u(c);const m=d,f=r.stringify(m,h);iT(f)?h.value=f:h.result=f,u(c,h)});function u(c,d){c||!d?a(c):o?o(d):n(void 0,d)}}}processSync(t){let n=!1,r;return this.freeze(),Js("processSync",this.parser||this.Parser),el("processSync",this.compiler||this.Compiler),this.process(t,i),nh("processSync","process",n),r;function i(o,a){n=!0,Xp(o),r=a}}run(t,n,r){th(t),this.freeze();const i=this.transformers;return!r&&typeof n=="function"&&(r=n,n=void 0),r?o(void 0,r):new Promise(o);function o(a,s){const l=Mo(n);i.run(t,l,u);function u(c,d,h){const m=d||t;c?s(c):a?a(m):r(void 0,m,h)}}}runSync(t,n){let r=!1,i;return this.run(t,n,o),nh("runSync","run",r),i;function o(a,s){Xp(a),i=s,r=!0}}stringify(t,n){this.freeze();const r=Mo(n),i=this.compiler||this.Compiler;return el("stringify",i),th(t),i(t,r)}use(t,...n){const r=this.attachers,i=this.namespace;if(tl("use",this.frozen),t!=null)if(typeof t=="function")l(t,n);else if(typeof t=="object")Array.isArray(t)?s(t):a(t);else throw new TypeError("Expected usable value, not `"+t+"`");return this;function o(u){if(typeof u=="function")l(u,[]);else if(typeof u=="object")if(Array.isArray(u)){const[c,...d]=u;l(c,d)}else a(u);else throw new TypeError("Expected usable value, not `"+u+"`")}function a(u){if(!("plugins"in u)&&!("settings"in u))throw new Error("Expected usable value but received an empty preset, which is probably a mistake: presets typically come with `plugins` and sometimes with `settings`, but this has neither");s(u.plugins),u.settings&&(i.settings=Ys(!0,i.settings,u.settings))}function s(u){let c=-1;if(u!=null)if(Array.isArray(u))for(;++c<u.length;){const d=u[c];o(d)}else throw new TypeError("Expected a list of plugins, not `"+u+"`")}function l(u,c){let d=-1,h=-1;for(;++d<r.length;)if(r[d][0]===u){h=d;break}if(h===-1)r.push([u,...c]);else if(c.length>0){let[m,...f]=c;const x=r[h][1];ku(x)&&ku(m)&&(m=Ys(!0,x,m)),r[h]=[u,m,...f]}}}}const nT=new Od().freeze();function Js(e,t){if(typeof t!="function")throw new TypeError("Cannot `"+e+"` without `parser`")}function el(e,t){if(typeof t!="function")throw new TypeError("Cannot `"+e+"` without `compiler`")}function tl(e,t){if(t)throw new Error("Cannot call `"+e+"` on a frozen processor.\nCreate a new processor first, by calling it: use `processor()` instead of `processor`.")}function th(e){if(!ku(e)||typeof e.type!="string")throw new TypeError("Expected node, got `"+e+"`")}function nh(e,t,n){if(!n)throw new Error("`"+e+"` finished async. Use `"+t+"` instead")}function Mo(e){return rT(e)?e:new V0(e)}function rT(e){return!!(e&&typeof e=="object"&&"message"in e&&"messages"in e)}function iT(e){return typeof e=="string"||oT(e)}function oT(e){return!!(e&&typeof e=="object"&&"byteLength"in e&&"byteOffset"in e)}const aT="https://github.com/remarkjs/react-markdown/blob/main/changelog.md",rh=[],ih={allowDangerousHtml:!0},sT=/^(https?|ircs?|mailto|xmpp)$/i,lT=[{from:"astPlugins",id:"remove-buggy-html-in-markdown-parser"},{from:"allowDangerousHtml",id:"remove-buggy-html-in-markdown-parser"},{from:"allowNode",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"allowElement"},{from:"allowedTypes",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"allowedElements"},{from:"disallowedTypes",id:"replace-allownode-allowedtypes-and-disallowedtypes",to:"disallowedElements"},{from:"escapeHtml",id:"remove-buggy-html-in-markdown-parser"},{from:"includeElementIndex",id:"#remove-includeelementindex"},{from:"includeNodeIndex",id:"change-includenodeindex-to-includeelementindex"},{from:"linkTarget",id:"remove-linktarget"},{from:"plugins",id:"change-plugins-to-remarkplugins",to:"remarkPlugins"},{from:"rawSourcePos",id:"#remove-rawsourcepos"},{from:"renderers",id:"change-renderers-to-components",to:"components"},{from:"source",id:"change-source-to-children",to:"children"},{from:"sourcePos",id:"#remove-sourcepos"},{from:"transformImageUri",id:"#add-urltransform",to:"urlTransform"},{from:"transformLinkUri",id:"#add-urltransform",to:"urlTransform"}];function B0(e){const t=uT(e),n=dT(e);return cT(t.runSync(t.parse(n),n),e)}function uT(e){const t=e.rehypePlugins||rh,n=e.remarkPlugins||rh,r=e.remarkRehypeOptions?{...e.remarkRehypeOptions,...ih}:ih;return nT().use(H2).use(n).use(qw,r).use(t)}function dT(e){const t=e.children||"",n=new V0;return typeof t=="string"&&(n.value=t),n}function cT(e,t){const n=t.allowedElements,r=t.allowElement,i=t.components,o=t.disallowedElements,a=t.skipHtml,s=t.unwrapDisallowed,l=t.urlTransform||pT;for(const c of lT)Object.hasOwn(t,c.from)&&(""+c.from+(c.to?"use `"+c.to+"` instead":"remove it")+aT+c.id,void 0);return t.className&&(e={type:"element",tagName:"div",properties:{className:t.className},children:e.type==="root"?e.children:[e]}),qd(e,u),Ev(e,{Fragment:p.Fragment,components:i,ignoreInvalidStyle:!0,jsx:p.jsx,jsxs:p.jsxs,passKeys:!0,passNode:!0});function u(c,d,h){if(c.type==="raw"&&h&&typeof d=="number")return a?h.children.splice(d,1):h.children[d]={type:"text",value:c.value},d;if(c.type==="element"){let m;for(m in Qs)if(Object.hasOwn(Qs,m)&&Object.hasOwn(c.properties,m)){const f=c.properties[m],x=Qs[m];(x===null||x.includes(c.tagName))&&(c.properties[m]=l(String(f||""),m,c))}}if(c.type==="element"){let m=n?!n.includes(c.tagName):o?o.includes(c.tagName):!1;if(!m&&r&&typeof d=="number"&&(m=!r(c,d,h)),m&&h&&typeof d=="number")return s&&c.children?h.children.splice(d,1,...c.children):h.children.splice(d,1),d}}}function pT(e){const t=e.indexOf(":"),n=e.indexOf("?"),r=e.indexOf("#"),i=e.indexOf("/");return t===-1||i!==-1&&t>i||n!==-1&&t>n||r!==-1&&t>r||sT.test(e.slice(0,t))?e:""}function oh(e,t){const n=String(e);if(typeof t!="string")throw new TypeError("Expected character");let r=0,i=n.indexOf(t);for(;i!==-1;)r++,i=n.indexOf(t,i+t.length);return r}function hT(e){if(typeof e!="string")throw new TypeError("Expected a string");return e.replace(/[|\\{}()[\]^$+*?.]/g,"\\$&").replace(/-/g,"\\x2d")}function mT(e,t,n){const i=ds((n||{}).ignore||[]),o=fT(t);let a=-1;for(;++a<o.length;)O0(e,"text",s);function s(u,c){let d=-1,h;for(;++d<c.length;){const m=c[d],f=h?h.children:void 0;if(i(m,f?f.indexOf(m):void 0,h))return;h=m}if(h)return l(u,c)}function l(u,c){const d=c[c.length-1],h=o[a][0],m=o[a][1];let f=0;const S=d.children.indexOf(u);let g=!1,y=[];h.lastIndex=0;let _=h.exec(u.value);for(;_;){const w=_.index,z={index:_.index,input:_.input,stack:[...c,u]};let k=m(..._,z);if(typeof k=="string"&&(k=k.length>0?{type:"text",value:k}:void 0),k===!1?h.lastIndex=w+1:(f!==w&&y.push({type:"text",value:u.value.slice(f,w)}),Array.isArray(k)?y.push(...k):k&&y.push(k),f=w+_[0].length,g=!0),!h.global)break;_=h.exec(u.value)}return g?(f<u.value.length&&y.push({type:"text",value:u.value.slice(f)}),d.children.splice(S,1,...y)):y=[u],S+y.length}}function fT(e){const t=[];if(!Array.isArray(e))throw new TypeError("Expected find and replace tuple or list of tuples");const n=!e[0]||Array.isArray(e[0])?e:[e];let r=-1;for(;++r<n.length;){const i=n[r];t.push([gT(i[0]),_T(i[1])])}return t}function gT(e){return typeof e=="string"?new RegExp(hT(e),"g"):e}function _T(e){return typeof e=="function"?e:function(){return e}}const nl="phrasing",rl=["autolink","link","image","label"];function yT(){return{transforms:[CT],enter:{literalAutolink:bT,literalAutolinkEmail:il,literalAutolinkHttp:il,literalAutolinkWww:il},exit:{literalAutolink:TT,literalAutolinkEmail:wT,literalAutolinkHttp:vT,literalAutolinkWww:kT}}}function xT(){return{unsafe:[{character:"@",before:"[+\\-.\\w]",after:"[\\-.\\w]",inConstruct:nl,notInConstruct:rl},{character:".",before:"[Ww]",after:"[\\-.\\w]",inConstruct:nl,notInConstruct:rl},{character:":",before:"[ps]",after:"\\/",inConstruct:nl,notInConstruct:rl}]}}function bT(e){this.enter({type:"link",title:null,url:"",children:[]},e)}function il(e){this.config.enter.autolinkProtocol.call(this,e)}function vT(e){this.config.exit.autolinkProtocol.call(this,e)}function kT(e){this.config.exit.data.call(this,e);const t=this.stack[this.stack.length-1];t.type,t.url="http://"+this.sliceSerialize(e)}function wT(e){this.config.exit.autolinkEmail.call(this,e)}function TT(e){this.exit(e)}function CT(e){mT(e,[[/(https?:\/\/|www(?=\.))([-.\w]+)([^ \t\r\n]*)/gi,ST],[new RegExp("(?<=^|\\s|\\p{P}|\\p{S})([-.\\w+]+)@([-\\w]+(?:\\.[-\\w]+)+)","gu"),zT]],{ignore:["link","linkReference"]})}function ST(e,t,n,r,i){let o="";if(!U0(i)||(/^w/i.test(t)&&(n=t+n,t="",o="http://"),!ET(n)))return!1;const a=PT(n+r);if(!a[0])return!1;const s={type:"link",title:null,url:o+t+a[0],children:[{type:"text",value:t+a[0]}]};return a[1]?[s,{type:"text",value:a[1]}]:s}function zT(e,t,n,r){return!U0(r,!0)||/[-\d_]$/.test(n)?!1:{type:"link",title:null,url:"mailto:"+t+"@"+n,children:[{type:"text",value:t+"@"+n}]}}function ET(e){const t=e.split(".");return!(t.length<2||t[t.length-1]&&(/_/.test(t[t.length-1])||!/[a-zA-Z\d]/.test(t[t.length-1]))||t[t.length-2]&&(/_/.test(t[t.length-2])||!/[a-zA-Z\d]/.test(t[t.length-2])))}function PT(e){const t=/[!"&'),.:;<>?\]}]+$/.exec(e);if(!t)return[e,void 0];e=e.slice(0,t.index);let n=t[0],r=n.indexOf(")");const i=oh(e,"(");let o=oh(e,")");for(;r!==-1&&i>o;)e+=n.slice(0,r+1),n=n.slice(r+1),r=n.indexOf(")"),o++;return[e,n]}function U0(e,t){const n=e.input.charCodeAt(e.index-1);return(e.index===0||or(n)||ss(n))&&(!t||n!==47)}H0.peek=MT;function NT(){this.buffer()}function AT(e){this.enter({type:"footnoteReference",identifier:"",label:""},e)}function LT(){this.buffer()}function jT(e){this.enter({type:"footnoteDefinition",identifier:"",label:"",children:[]},e)}function RT(e){const t=this.resume(),n=this.stack[this.stack.length-1];n.type,n.identifier=Ot(this.sliceSerialize(e)).toLowerCase(),n.label=t}function FT(e){this.exit(e)}function IT(e){const t=this.resume(),n=this.stack[this.stack.length-1];n.type,n.identifier=Ot(this.sliceSerialize(e)).toLowerCase(),n.label=t}function DT(e){this.exit(e)}function MT(){return"["}function H0(e,t,n,r){const i=n.createTracker(r);let o=i.move("[^");const a=n.enter("footnoteReference"),s=n.enter("reference");return o+=i.move(n.safe(n.associationId(e),{after:"]",before:o})),s(),a(),o+=i.move("]"),o}function qT(){return{enter:{gfmFootnoteCallString:NT,gfmFootnoteCall:AT,gfmFootnoteDefinitionLabelString:LT,gfmFootnoteDefinition:jT},exit:{gfmFootnoteCallString:RT,gfmFootnoteCall:FT,gfmFootnoteDefinitionLabelString:IT,gfmFootnoteDefinition:DT}}}function OT(e){let t=!1;return e&&e.firstLineBlank&&(t=!0),{handlers:{footnoteDefinition:n,footnoteReference:H0},unsafe:[{character:"[",inConstruct:["label","phrasing","reference"]}]};function n(r,i,o,a){const s=o.createTracker(a);let l=s.move("[^");const u=o.enter("footnoteDefinition"),c=o.enter("label");return l+=s.move(o.safe(o.associationId(r),{before:l,after:"]"})),c(),l+=s.move("]:"),r.children&&r.children.length>0&&(s.shift(4),l+=s.move((t?`
`:" ")+o.indentLines(o.containerFlow(r,s.current()),t?Q0:WT))),u(),l}}function WT(e,t,n){return t===0?e:Q0(e,t,n)}function Q0(e,t,n){return(n?"":"    ")+e}const VT=["autolink","destinationLiteral","destinationRaw","reference","titleQuote","titleApostrophe"];X0.peek=XT;function BT(){return{canContainEols:["delete"],enter:{strikethrough:HT},exit:{strikethrough:QT}}}function UT(){return{unsafe:[{character:"~",inConstruct:"phrasing",notInConstruct:VT}],handlers:{delete:X0}}}function HT(e){this.enter({type:"delete",children:[]},e)}function QT(e){this.exit(e)}function X0(e,t,n,r){const i=n.createTracker(r),o=n.enter("strikethrough");let a=i.move("~~");return a+=n.containerPhrasing(e,{...i.current(),before:a,after:"~"}),a+=i.move("~~"),o(),a}function XT(){return"~"}function KT(e){return e.length}function YT(e,t){const n=t||{},r=(n.align||[]).concat(),i=n.stringLength||KT,o=[],a=[],s=[],l=[];let u=0,c=-1;for(;++c<e.length;){const x=[],S=[];let g=-1;for(e[c].length>u&&(u=e[c].length);++g<e[c].length;){const y=$T(e[c][g]);if(n.alignDelimiters!==!1){const _=i(y);S[g]=_,(l[g]===void 0||_>l[g])&&(l[g]=_)}x.push(y)}a[c]=x,s[c]=S}let d=-1;if(typeof r=="object"&&"length"in r)for(;++d<u;)o[d]=ah(r[d]);else{const x=ah(r);for(;++d<u;)o[d]=x}d=-1;const h=[],m=[];for(;++d<u;){const x=o[d];let S="",g="";x===99?(S=":",g=":"):x===108?S=":":x===114&&(g=":");let y=n.alignDelimiters===!1?1:Math.max(1,l[d]-S.length-g.length);const _=S+"-".repeat(y)+g;n.alignDelimiters!==!1&&(y=S.length+y+g.length,y>l[d]&&(l[d]=y),m[d]=y),h[d]=_}a.splice(1,0,h),s.splice(1,0,m),c=-1;const f=[];for(;++c<a.length;){const x=a[c],S=s[c];d=-1;const g=[];for(;++d<u;){const y=x[d]||"";let _="",w="";if(n.alignDelimiters!==!1){const z=l[d]-(S[d]||0),k=o[d];k===114?_=" ".repeat(z):k===99?z%2?(_=" ".repeat(z/2+.5),w=" ".repeat(z/2-.5)):(_=" ".repeat(z/2),w=_):w=" ".repeat(z)}n.delimiterStart!==!1&&!d&&g.push("|"),n.padding!==!1&&!(n.alignDelimiters===!1&&y==="")&&(n.delimiterStart!==!1||d)&&g.push(" "),n.alignDelimiters!==!1&&g.push(_),g.push(y),n.alignDelimiters!==!1&&g.push(w),n.padding!==!1&&g.push(" "),(n.delimiterEnd!==!1||d!==u-1)&&g.push("|")}f.push(n.delimiterEnd===!1?g.join("").replace(/ +$/,""):g.join(""))}return f.join(`
`)}function $T(e){return e==null?"":String(e)}function ah(e){const t=typeof e=="string"?e.codePointAt(0):0;return t===67||t===99?99:t===76||t===108?108:t===82||t===114?114:0}function GT(e,t,n,r){const i=n.enter("blockquote"),o=n.createTracker(r);o.move("> "),o.shift(2);const a=n.indentLines(n.containerFlow(e,o.current()),ZT);return i(),a}function ZT(e,t,n){return">"+(n?"":" ")+e}function JT(e,t){return sh(e,t.inConstruct,!0)&&!sh(e,t.notInConstruct,!1)}function sh(e,t,n){if(typeof t=="string"&&(t=[t]),!t||t.length===0)return n;let r=-1;for(;++r<t.length;)if(e.includes(t[r]))return!0;return!1}function lh(e,t,n,r){let i=-1;for(;++i<n.unsafe.length;)if(n.unsafe[i].character===`
`&&JT(n.stack,n.unsafe[i]))return/[ \t]/.test(r.before)?"":" ";return`\\
`}function eC(e,t){const n=String(e);let r=n.indexOf(t),i=r,o=0,a=0;if(typeof t!="string")throw new TypeError("Expected substring");for(;r!==-1;)r===i?++o>a&&(a=o):o=1,i=r+t.length,r=n.indexOf(t,i);return a}function tC(e,t){return!!(t.options.fences===!1&&e.value&&!e.lang&&/[^ \r\n]/.test(e.value)&&!/^[\t ]*(?:[\r\n]|$)|(?:^|[\r\n])[\t ]*$/.test(e.value))}function nC(e){const t=e.options.fence||"`";if(t!=="`"&&t!=="~")throw new Error("Cannot serialize code with `"+t+"` for `options.fence`, expected `` ` `` or `~`");return t}function rC(e,t,n,r){const i=nC(n),o=e.value||"",a=i==="`"?"GraveAccent":"Tilde";if(tC(e,n)){const d=n.enter("codeIndented"),h=n.indentLines(o,iC);return d(),h}const s=n.createTracker(r),l=i.repeat(Math.max(eC(o,i)+1,3)),u=n.enter("codeFenced");let c=s.move(l);if(e.lang){const d=n.enter(`codeFencedLang${a}`);c+=s.move(n.safe(e.lang,{before:c,after:" ",encode:["`"],...s.current()})),d()}if(e.lang&&e.meta){const d=n.enter(`codeFencedMeta${a}`);c+=s.move(" "),c+=s.move(n.safe(e.meta,{before:c,after:`
`,encode:["`"],...s.current()})),d()}return c+=s.move(`
`),o&&(c+=s.move(o+`
`)),c+=s.move(l),u(),c}function iC(e,t,n){return(n?"":"    ")+e}function Wd(e){const t=e.options.quote||'"';if(t!=='"'&&t!=="'")throw new Error("Cannot serialize title with `"+t+"` for `options.quote`, expected `\"`, or `'`");return t}function oC(e,t,n,r){const i=Wd(n),o=i==='"'?"Quote":"Apostrophe",a=n.enter("definition");let s=n.enter("label");const l=n.createTracker(r);let u=l.move("[");return u+=l.move(n.safe(n.associationId(e),{before:u,after:"]",...l.current()})),u+=l.move("]: "),s(),!e.url||/[\0- \u007F]/.test(e.url)?(s=n.enter("destinationLiteral"),u+=l.move("<"),u+=l.move(n.safe(e.url,{before:u,after:">",...l.current()})),u+=l.move(">")):(s=n.enter("destinationRaw"),u+=l.move(n.safe(e.url,{before:u,after:e.title?" ":`
`,...l.current()}))),s(),e.title&&(s=n.enter(`title${o}`),u+=l.move(" "+i),u+=l.move(n.safe(e.title,{before:u,after:i,...l.current()})),u+=l.move(i),s()),a(),u}function aC(e){const t=e.options.emphasis||"*";if(t!=="*"&&t!=="_")throw new Error("Cannot serialize emphasis with `"+t+"` for `options.emphasis`, expected `*`, or `_`");return t}function eo(e){return"&#x"+e.toString(16).toUpperCase()+";"}function Ma(e,t,n){const r=Kr(e),i=Kr(t);return r===void 0?i===void 0?n==="_"?{inside:!0,outside:!0}:{inside:!1,outside:!1}:i===1?{inside:!0,outside:!0}:{inside:!1,outside:!0}:r===1?i===void 0?{inside:!1,outside:!1}:i===1?{inside:!0,outside:!0}:{inside:!1,outside:!1}:i===void 0?{inside:!1,outside:!1}:i===1?{inside:!0,outside:!1}:{inside:!1,outside:!1}}K0.peek=sC;function K0(e,t,n,r){const i=aC(n),o=n.enter("emphasis"),a=n.createTracker(r),s=a.move(i);let l=a.move(n.containerPhrasing(e,{after:i,before:s,...a.current()}));const u=l.charCodeAt(0),c=Ma(r.before.charCodeAt(r.before.length-1),u,i);c.inside&&(l=eo(u)+l.slice(1));const d=l.charCodeAt(l.length-1),h=Ma(r.after.charCodeAt(0),d,i);h.inside&&(l=l.slice(0,-1)+eo(d));const m=a.move(i);return o(),n.attentionEncodeSurroundingInfo={after:h.outside,before:c.outside},s+l+m}function sC(e,t,n){return n.options.emphasis||"*"}function lC(e,t){let n=!1;return qd(e,function(r){if("value"in r&&/\r?\n|\r/.test(r.value)||r.type==="break")return n=!0,bu}),!!((!e.depth||e.depth<3)&&Ld(e)&&(t.options.setext||n))}function uC(e,t,n,r){const i=Math.max(Math.min(6,e.depth||1),1),o=n.createTracker(r);if(lC(e,n)){const c=n.enter("headingSetext"),d=n.enter("phrasing"),h=n.containerPhrasing(e,{...o.current(),before:`
`,after:`
`});return d(),c(),h+`
`+(i===1?"=":"-").repeat(h.length-(Math.max(h.lastIndexOf("\r"),h.lastIndexOf(`
`))+1))}const a="#".repeat(i),s=n.enter("headingAtx"),l=n.enter("phrasing");o.move(a+" ");let u=n.containerPhrasing(e,{before:"# ",after:`
`,...o.current()});return/^[\t ]/.test(u)&&(u=eo(u.charCodeAt(0))+u.slice(1)),u=u?a+" "+u:a,n.options.closeAtx&&(u+=" "+a),l(),s(),u}Y0.peek=dC;function Y0(e){return e.value||""}function dC(){return"<"}$0.peek=cC;function $0(e,t,n,r){const i=Wd(n),o=i==='"'?"Quote":"Apostrophe",a=n.enter("image");let s=n.enter("label");const l=n.createTracker(r);let u=l.move("![");return u+=l.move(n.safe(e.alt,{before:u,after:"]",...l.current()})),u+=l.move("]("),s(),!e.url&&e.title||/[\0- \u007F]/.test(e.url)?(s=n.enter("destinationLiteral"),u+=l.move("<"),u+=l.move(n.safe(e.url,{before:u,after:">",...l.current()})),u+=l.move(">")):(s=n.enter("destinationRaw"),u+=l.move(n.safe(e.url,{before:u,after:e.title?" ":")",...l.current()}))),s(),e.title&&(s=n.enter(`title${o}`),u+=l.move(" "+i),u+=l.move(n.safe(e.title,{before:u,after:i,...l.current()})),u+=l.move(i),s()),u+=l.move(")"),a(),u}function cC(){return"!"}G0.peek=pC;function G0(e,t,n,r){const i=e.referenceType,o=n.enter("imageReference");let a=n.enter("label");const s=n.createTracker(r);let l=s.move("![");const u=n.safe(e.alt,{before:l,after:"]",...s.current()});l+=s.move(u+"]["),a();const c=n.stack;n.stack=[],a=n.enter("reference");const d=n.safe(n.associationId(e),{before:l,after:"]",...s.current()});return a(),n.stack=c,o(),i==="full"||!u||u!==d?l+=s.move(d+"]"):i==="shortcut"?l=l.slice(0,-1):l+=s.move("]"),l}function pC(){return"!"}Z0.peek=hC;function Z0(e,t,n){let r=e.value||"",i="`",o=-1;for(;new RegExp("(^|[^`])"+i+"([^`]|$)").test(r);)i+="`";for(/[^ \r\n]/.test(r)&&(/^[ \r\n]/.test(r)&&/[ \r\n]$/.test(r)||/^`|`$/.test(r))&&(r=" "+r+" ");++o<n.unsafe.length;){const a=n.unsafe[o],s=n.compilePattern(a);let l;if(a.atBreak)for(;l=s.exec(r);){let u=l.index;r.charCodeAt(u)===10&&r.charCodeAt(u-1)===13&&u--,r=r.slice(0,u)+" "+r.slice(l.index+1)}}return i+r+i}function hC(){return"`"}function J0(e,t){const n=Ld(e);return!!(!t.options.resourceLink&&e.url&&!e.title&&e.children&&e.children.length===1&&e.children[0].type==="text"&&(n===e.url||"mailto:"+n===e.url)&&/^[a-z][a-z+.-]+:/i.test(e.url)&&!/[\0- <>\u007F]/.test(e.url))}eg.peek=mC;function eg(e,t,n,r){const i=Wd(n),o=i==='"'?"Quote":"Apostrophe",a=n.createTracker(r);let s,l;if(J0(e,n)){const c=n.stack;n.stack=[],s=n.enter("autolink");let d=a.move("<");return d+=a.move(n.containerPhrasing(e,{before:d,after:">",...a.current()})),d+=a.move(">"),s(),n.stack=c,d}s=n.enter("link"),l=n.enter("label");let u=a.move("[");return u+=a.move(n.containerPhrasing(e,{before:u,after:"](",...a.current()})),u+=a.move("]("),l(),!e.url&&e.title||/[\0- \u007F]/.test(e.url)?(l=n.enter("destinationLiteral"),u+=a.move("<"),u+=a.move(n.safe(e.url,{before:u,after:">",...a.current()})),u+=a.move(">")):(l=n.enter("destinationRaw"),u+=a.move(n.safe(e.url,{before:u,after:e.title?" ":")",...a.current()}))),l(),e.title&&(l=n.enter(`title${o}`),u+=a.move(" "+i),u+=a.move(n.safe(e.title,{before:u,after:i,...a.current()})),u+=a.move(i),l()),u+=a.move(")"),s(),u}function mC(e,t,n){return J0(e,n)?"<":"["}tg.peek=fC;function tg(e,t,n,r){const i=e.referenceType,o=n.enter("linkReference");let a=n.enter("label");const s=n.createTracker(r);let l=s.move("[");const u=n.containerPhrasing(e,{before:l,after:"]",...s.current()});l+=s.move(u+"]["),a();const c=n.stack;n.stack=[],a=n.enter("reference");const d=n.safe(n.associationId(e),{before:l,after:"]",...s.current()});return a(),n.stack=c,o(),i==="full"||!u||u!==d?l+=s.move(d+"]"):i==="shortcut"?l=l.slice(0,-1):l+=s.move("]"),l}function fC(){return"["}function Vd(e){const t=e.options.bullet||"*";if(t!=="*"&&t!=="+"&&t!=="-")throw new Error("Cannot serialize items with `"+t+"` for `options.bullet`, expected `*`, `+`, or `-`");return t}function gC(e){const t=Vd(e),n=e.options.bulletOther;if(!n)return t==="*"?"-":"*";if(n!=="*"&&n!=="+"&&n!=="-")throw new Error("Cannot serialize items with `"+n+"` for `options.bulletOther`, expected `*`, `+`, or `-`");if(n===t)throw new Error("Expected `bullet` (`"+t+"`) and `bulletOther` (`"+n+"`) to be different");return n}function _C(e){const t=e.options.bulletOrdered||".";if(t!=="."&&t!==")")throw new Error("Cannot serialize items with `"+t+"` for `options.bulletOrdered`, expected `.` or `)`");return t}function ng(e){const t=e.options.rule||"*";if(t!=="*"&&t!=="-"&&t!=="_")throw new Error("Cannot serialize rules with `"+t+"` for `options.rule`, expected `*`, `-`, or `_`");return t}function yC(e,t,n,r){const i=n.enter("list"),o=n.bulletCurrent;let a=e.ordered?_C(n):Vd(n);const s=e.ordered?a==="."?")":".":gC(n);let l=t&&n.bulletLastUsed?a===n.bulletLastUsed:!1;if(!e.ordered){const c=e.children?e.children[0]:void 0;if((a==="*"||a==="-")&&c&&(!c.children||!c.children[0])&&n.stack[n.stack.length-1]==="list"&&n.stack[n.stack.length-2]==="listItem"&&n.stack[n.stack.length-3]==="list"&&n.stack[n.stack.length-4]==="listItem"&&n.indexStack[n.indexStack.length-1]===0&&n.indexStack[n.indexStack.length-2]===0&&n.indexStack[n.indexStack.length-3]===0&&(l=!0),ng(n)===a&&c){let d=-1;for(;++d<e.children.length;){const h=e.children[d];if(h&&h.type==="listItem"&&h.children&&h.children[0]&&h.children[0].type==="thematicBreak"){l=!0;break}}}}l&&(a=s),n.bulletCurrent=a;const u=n.containerFlow(e,r);return n.bulletLastUsed=a,n.bulletCurrent=o,i(),u}function xC(e){const t=e.options.listItemIndent||"one";if(t!=="tab"&&t!=="one"&&t!=="mixed")throw new Error("Cannot serialize items with `"+t+"` for `options.listItemIndent`, expected `tab`, `one`, or `mixed`");return t}function bC(e,t,n,r){const i=xC(n);let o=n.bulletCurrent||Vd(n);t&&t.type==="list"&&t.ordered&&(o=(typeof t.start=="number"&&t.start>-1?t.start:1)+(n.options.incrementListMarker===!1?0:t.children.indexOf(e))+o);let a=o.length+1;(i==="tab"||i==="mixed"&&(t&&t.type==="list"&&t.spread||e.spread))&&(a=Math.ceil(a/4)*4);const s=n.createTracker(r);s.move(o+" ".repeat(a-o.length)),s.shift(a);const l=n.enter("listItem"),u=n.indentLines(n.containerFlow(e,s.current()),c);return l(),u;function c(d,h,m){return h?(m?"":" ".repeat(a))+d:(m?o:o+" ".repeat(a-o.length))+d}}function vC(e,t,n,r){const i=n.enter("paragraph"),o=n.enter("phrasing"),a=n.containerPhrasing(e,r);return o(),i(),a}const kC=ds(["break","delete","emphasis","footnote","footnoteReference","image","imageReference","inlineCode","inlineMath","link","linkReference","mdxJsxTextElement","mdxTextExpression","strong","text","textDirective"]);function wC(e,t,n,r){return(e.children.some(function(a){return kC(a)})?n.containerPhrasing:n.containerFlow).call(n,e,r)}function TC(e){const t=e.options.strong||"*";if(t!=="*"&&t!=="_")throw new Error("Cannot serialize strong with `"+t+"` for `options.strong`, expected `*`, or `_`");return t}rg.peek=CC;function rg(e,t,n,r){const i=TC(n),o=n.enter("strong"),a=n.createTracker(r),s=a.move(i+i);let l=a.move(n.containerPhrasing(e,{after:i,before:s,...a.current()}));const u=l.charCodeAt(0),c=Ma(r.before.charCodeAt(r.before.length-1),u,i);c.inside&&(l=eo(u)+l.slice(1));const d=l.charCodeAt(l.length-1),h=Ma(r.after.charCodeAt(0),d,i);h.inside&&(l=l.slice(0,-1)+eo(d));const m=a.move(i+i);return o(),n.attentionEncodeSurroundingInfo={after:h.outside,before:c.outside},s+l+m}function CC(e,t,n){return n.options.strong||"*"}function SC(e,t,n,r){return n.safe(e.value,r)}function zC(e){const t=e.options.ruleRepetition||3;if(t<3)throw new Error("Cannot serialize rules with repetition `"+t+"` for `options.ruleRepetition`, expected `3` or more");return t}function EC(e,t,n){const r=(ng(n)+(n.options.ruleSpaces?" ":"")).repeat(zC(n));return n.options.ruleSpaces?r.slice(0,-1):r}const ig={blockquote:GT,break:lh,code:rC,definition:oC,emphasis:K0,hardBreak:lh,heading:uC,html:Y0,image:$0,imageReference:G0,inlineCode:Z0,link:eg,linkReference:tg,list:yC,listItem:bC,paragraph:vC,root:wC,strong:rg,text:SC,thematicBreak:EC};function PC(){return{enter:{table:NC,tableData:uh,tableHeader:uh,tableRow:LC},exit:{codeText:jC,table:AC,tableData:ol,tableHeader:ol,tableRow:ol}}}function NC(e){const t=e._align;this.enter({type:"table",align:t.map(function(n){return n==="none"?null:n}),children:[]},e),this.data.inTable=!0}function AC(e){this.exit(e),this.data.inTable=void 0}function LC(e){this.enter({type:"tableRow",children:[]},e)}function ol(e){this.exit(e)}function uh(e){this.enter({type:"tableCell",children:[]},e)}function jC(e){let t=this.resume();this.data.inTable&&(t=t.replace(/\\([\\|])/g,RC));const n=this.stack[this.stack.length-1];n.type,n.value=t,this.exit(e)}function RC(e,t){return t==="|"?t:e}function FC(e){const t=e||{},n=t.tableCellPadding,r=t.tablePipeAlign,i=t.stringLength,o=n?" ":"|";return{unsafe:[{character:"\r",inConstruct:"tableCell"},{character:`
`,inConstruct:"tableCell"},{atBreak:!0,character:"|",after:"[	 :-]"},{character:"|",inConstruct:"tableCell"},{atBreak:!0,character:":",after:"-"},{atBreak:!0,character:"-",after:"[:|-]"}],handlers:{inlineCode:h,table:a,tableCell:l,tableRow:s}};function a(m,f,x,S){return u(c(m,x,S),m.align)}function s(m,f,x,S){const g=d(m,x,S),y=u([g]);return y.slice(0,y.indexOf(`
`))}function l(m,f,x,S){const g=x.enter("tableCell"),y=x.enter("phrasing"),_=x.containerPhrasing(m,{...S,before:o,after:o});return y(),g(),_}function u(m,f){return YT(m,{align:f,alignDelimiters:r,padding:n,stringLength:i})}function c(m,f,x){const S=m.children;let g=-1;const y=[],_=f.enter("table");for(;++g<S.length;)y[g]=d(S[g],f,x);return _(),y}function d(m,f,x){const S=m.children;let g=-1;const y=[],_=f.enter("tableRow");for(;++g<S.length;)y[g]=l(S[g],m,f,x);return _(),y}function h(m,f,x){let S=ig.inlineCode(m,f,x);return x.stack.includes("tableCell")&&(S=S.replace(/\|/g,"\\$&")),S}}function IC(){return{exit:{taskListCheckValueChecked:dh,taskListCheckValueUnchecked:dh,paragraph:MC}}}function DC(){return{unsafe:[{atBreak:!0,character:"-",after:"[:|-]"}],handlers:{listItem:qC}}}function dh(e){const t=this.stack[this.stack.length-2];t.type,t.checked=e.type==="taskListCheckValueChecked"}function MC(e){const t=this.stack[this.stack.length-2];if(t&&t.type==="listItem"&&typeof t.checked=="boolean"){const n=this.stack[this.stack.length-1];n.type;const r=n.children[0];if(r&&r.type==="text"){const i=t.children;let o=-1,a;for(;++o<i.length;){const s=i[o];if(s.type==="paragraph"){a=s;break}}a===n&&(r.value=r.value.slice(1),r.value.length===0?n.children.shift():n.position&&r.position&&typeof r.position.start.offset=="number"&&(r.position.start.column++,r.position.start.offset++,n.position.start=Object.assign({},r.position.start)))}}this.exit(e)}function qC(e,t,n,r){const i=e.children[0],o=typeof e.checked=="boolean"&&i&&i.type==="paragraph",a="["+(e.checked?"x":" ")+"] ",s=n.createTracker(r);o&&s.move(a);let l=ig.listItem(e,t,n,{...r,...s.current()});return o&&(l=l.replace(/^(?:[*+-]|\d+\.)([\r\n]| {1,3})/,u)),l;function u(c){return c+a}}function OC(){return[yT(),qT(),BT(),PC(),IC()]}function WC(e){return{extensions:[xT(),OT(e),UT(),FC(e),DC()]}}const VC={tokenize:KC,partial:!0},og={tokenize:YC,partial:!0},ag={tokenize:$C,partial:!0},sg={tokenize:GC,partial:!0},BC={tokenize:ZC,partial:!0},lg={name:"wwwAutolink",tokenize:QC,previous:dg},ug={name:"protocolAutolink",tokenize:XC,previous:cg},pn={name:"emailAutolink",tokenize:HC,previous:pg},Yt={};function UC(){return{text:Yt}}let Bn=48;for(;Bn<123;)Yt[Bn]=pn,Bn++,Bn===58?Bn=65:Bn===91&&(Bn=97);Yt[43]=pn;Yt[45]=pn;Yt[46]=pn;Yt[95]=pn;Yt[72]=[pn,ug];Yt[104]=[pn,ug];Yt[87]=[pn,lg];Yt[119]=[pn,lg];function HC(e,t,n){const r=this;let i,o;return a;function a(d){return!Tu(d)||!pg.call(r,r.previous)||Bd(r.events)?n(d):(e.enter("literalAutolink"),e.enter("literalAutolinkEmail"),s(d))}function s(d){return Tu(d)?(e.consume(d),s):d===64?(e.consume(d),l):n(d)}function l(d){return d===46?e.check(BC,c,u)(d):d===45||d===95||Ue(d)?(o=!0,e.consume(d),l):c(d)}function u(d){return e.consume(d),i=!0,l}function c(d){return o&&i&&$e(r.previous)?(e.exit("literalAutolinkEmail"),e.exit("literalAutolink"),t(d)):n(d)}}function QC(e,t,n){const r=this;return i;function i(a){return a!==87&&a!==119||!dg.call(r,r.previous)||Bd(r.events)?n(a):(e.enter("literalAutolink"),e.enter("literalAutolinkWww"),e.check(VC,e.attempt(og,e.attempt(ag,o),n),n)(a))}function o(a){return e.exit("literalAutolinkWww"),e.exit("literalAutolink"),t(a)}}function XC(e,t,n){const r=this;let i="",o=!1;return a;function a(d){return(d===72||d===104)&&cg.call(r,r.previous)&&!Bd(r.events)?(e.enter("literalAutolink"),e.enter("literalAutolinkHttp"),i+=String.fromCodePoint(d),e.consume(d),s):n(d)}function s(d){if($e(d)&&i.length<5)return i+=String.fromCodePoint(d),e.consume(d),s;if(d===58){const h=i.toLowerCase();if(h==="http"||h==="https")return e.consume(d),l}return n(d)}function l(d){return d===47?(e.consume(d),o?u:(o=!0,l)):n(d)}function u(d){return d===null||Fa(d)||pe(d)||or(d)||ss(d)?n(d):e.attempt(og,e.attempt(ag,c),n)(d)}function c(d){return e.exit("literalAutolinkHttp"),e.exit("literalAutolink"),t(d)}}function KC(e,t,n){let r=0;return i;function i(a){return(a===87||a===119)&&r<3?(r++,e.consume(a),i):a===46&&r===3?(e.consume(a),o):n(a)}function o(a){return a===null?n(a):t(a)}}function YC(e,t,n){let r,i,o;return a;function a(u){return u===46||u===95?e.check(sg,l,s)(u):u===null||pe(u)||or(u)||u!==45&&ss(u)?l(u):(o=!0,e.consume(u),a)}function s(u){return u===95?r=!0:(i=r,r=void 0),e.consume(u),a}function l(u){return i||r||!o?n(u):t(u)}}function $C(e,t){let n=0,r=0;return i;function i(a){return a===40?(n++,e.consume(a),i):a===41&&r<n?o(a):a===33||a===34||a===38||a===39||a===41||a===42||a===44||a===46||a===58||a===59||a===60||a===63||a===93||a===95||a===126?e.check(sg,t,o)(a):a===null||pe(a)||or(a)?t(a):(e.consume(a),i)}function o(a){return a===41&&r++,e.consume(a),i}}function GC(e,t,n){return r;function r(s){return s===33||s===34||s===39||s===41||s===42||s===44||s===46||s===58||s===59||s===63||s===95||s===126?(e.consume(s),r):s===38?(e.consume(s),o):s===93?(e.consume(s),i):s===60||s===null||pe(s)||or(s)?t(s):n(s)}function i(s){return s===null||s===40||s===91||pe(s)||or(s)?t(s):r(s)}function o(s){return $e(s)?a(s):n(s)}function a(s){return s===59?(e.consume(s),r):$e(s)?(e.consume(s),a):n(s)}}function ZC(e,t,n){return r;function r(o){return e.consume(o),i}function i(o){return Ue(o)?n(o):t(o)}}function dg(e){return e===null||e===40||e===42||e===95||e===91||e===93||e===126||pe(e)}function cg(e){return!$e(e)}function pg(e){return!(e===47||Tu(e))}function Tu(e){return e===43||e===45||e===46||e===95||Ue(e)}function Bd(e){let t=e.length,n=!1;for(;t--;){const r=e[t][1];if((r.type==="labelLink"||r.type==="labelImage")&&!r._balanced){n=!0;break}if(r._gfmAutolinkLiteralWalkedInto){n=!1;break}}return e.length>0&&!n&&(e[e.length-1][1]._gfmAutolinkLiteralWalkedInto=!0),n}const JC={tokenize:s4,partial:!0};function e4(){return{document:{91:{name:"gfmFootnoteDefinition",tokenize:i4,continuation:{tokenize:o4},exit:a4}},text:{91:{name:"gfmFootnoteCall",tokenize:r4},93:{name:"gfmPotentialFootnoteCall",add:"after",tokenize:t4,resolveTo:n4}}}}function t4(e,t,n){const r=this;let i=r.events.length;const o=r.parser.gfmFootnotes||(r.parser.gfmFootnotes=[]);let a;for(;i--;){const l=r.events[i][1];if(l.type==="labelImage"){a=l;break}if(l.type==="gfmFootnoteCall"||l.type==="labelLink"||l.type==="label"||l.type==="image"||l.type==="link")break}return s;function s(l){if(!a||!a._balanced)return n(l);const u=Ot(r.sliceSerialize({start:a.end,end:r.now()}));return u.codePointAt(0)!==94||!o.includes(u.slice(1))?n(l):(e.enter("gfmFootnoteCallLabelMarker"),e.consume(l),e.exit("gfmFootnoteCallLabelMarker"),t(l))}}function n4(e,t){let n=e.length;for(;n--;)if(e[n][1].type==="labelImage"&&e[n][0]==="enter"){e[n][1];break}e[n+1][1].type="data",e[n+3][1].type="gfmFootnoteCallLabelMarker";const r={type:"gfmFootnoteCall",start:Object.assign({},e[n+3][1].start),end:Object.assign({},e[e.length-1][1].end)},i={type:"gfmFootnoteCallMarker",start:Object.assign({},e[n+3][1].end),end:Object.assign({},e[n+3][1].end)};i.end.column++,i.end.offset++,i.end._bufferIndex++;const o={type:"gfmFootnoteCallString",start:Object.assign({},i.end),end:Object.assign({},e[e.length-1][1].start)},a={type:"chunkString",contentType:"string",start:Object.assign({},o.start),end:Object.assign({},o.end)},s=[e[n+1],e[n+2],["enter",r,t],e[n+3],e[n+4],["enter",i,t],["exit",i,t],["enter",o,t],["enter",a,t],["exit",a,t],["exit",o,t],e[e.length-2],e[e.length-1],["exit",r,t]];return e.splice(n,e.length-n+1,...s),e}function r4(e,t,n){const r=this,i=r.parser.gfmFootnotes||(r.parser.gfmFootnotes=[]);let o=0,a;return s;function s(d){return e.enter("gfmFootnoteCall"),e.enter("gfmFootnoteCallLabelMarker"),e.consume(d),e.exit("gfmFootnoteCallLabelMarker"),l}function l(d){return d!==94?n(d):(e.enter("gfmFootnoteCallMarker"),e.consume(d),e.exit("gfmFootnoteCallMarker"),e.enter("gfmFootnoteCallString"),e.enter("chunkString").contentType="string",u)}function u(d){if(o>999||d===93&&!a||d===null||d===91||pe(d))return n(d);if(d===93){e.exit("chunkString");const h=e.exit("gfmFootnoteCallString");return i.includes(Ot(r.sliceSerialize(h)))?(e.enter("gfmFootnoteCallLabelMarker"),e.consume(d),e.exit("gfmFootnoteCallLabelMarker"),e.exit("gfmFootnoteCall"),t):n(d)}return pe(d)||(a=!0),o++,e.consume(d),d===92?c:u}function c(d){return d===91||d===92||d===93?(e.consume(d),o++,u):u(d)}}function i4(e,t,n){const r=this,i=r.parser.gfmFootnotes||(r.parser.gfmFootnotes=[]);let o,a=0,s;return l;function l(f){return e.enter("gfmFootnoteDefinition")._container=!0,e.enter("gfmFootnoteDefinitionLabel"),e.enter("gfmFootnoteDefinitionLabelMarker"),e.consume(f),e.exit("gfmFootnoteDefinitionLabelMarker"),u}function u(f){return f===94?(e.enter("gfmFootnoteDefinitionMarker"),e.consume(f),e.exit("gfmFootnoteDefinitionMarker"),e.enter("gfmFootnoteDefinitionLabelString"),e.enter("chunkString").contentType="string",c):n(f)}function c(f){if(a>999||f===93&&!s||f===null||f===91||pe(f))return n(f);if(f===93){e.exit("chunkString");const x=e.exit("gfmFootnoteDefinitionLabelString");return o=Ot(r.sliceSerialize(x)),e.enter("gfmFootnoteDefinitionLabelMarker"),e.consume(f),e.exit("gfmFootnoteDefinitionLabelMarker"),e.exit("gfmFootnoteDefinitionLabel"),h}return pe(f)||(s=!0),a++,e.consume(f),f===92?d:c}function d(f){return f===91||f===92||f===93?(e.consume(f),a++,c):c(f)}function h(f){return f===58?(e.enter("definitionMarker"),e.consume(f),e.exit("definitionMarker"),i.includes(o)||i.push(o),oe(e,m,"gfmFootnoteDefinitionWhitespace")):n(f)}function m(f){return t(f)}}function o4(e,t,n){return e.check(co,t,e.attempt(JC,t,n))}function a4(e){e.exit("gfmFootnoteDefinition")}function s4(e,t,n){const r=this;return oe(e,i,"gfmFootnoteDefinitionIndent",5);function i(o){const a=r.events[r.events.length-1];return a&&a[1].type==="gfmFootnoteDefinitionIndent"&&a[2].sliceSerialize(a[1],!0).length===4?t(o):n(o)}}function l4(e){let n=(e||{}).singleTilde;const r={name:"strikethrough",tokenize:o,resolveAll:i};return n==null&&(n=!0),{text:{126:r},insideSpan:{null:[r]},attentionMarkers:{null:[126]}};function i(a,s){let l=-1;for(;++l<a.length;)if(a[l][0]==="enter"&&a[l][1].type==="strikethroughSequenceTemporary"&&a[l][1]._close){let u=l;for(;u--;)if(a[u][0]==="exit"&&a[u][1].type==="strikethroughSequenceTemporary"&&a[u][1]._open&&a[l][1].end.offset-a[l][1].start.offset===a[u][1].end.offset-a[u][1].start.offset){a[l][1].type="strikethroughSequence",a[u][1].type="strikethroughSequence";const c={type:"strikethrough",start:Object.assign({},a[u][1].start),end:Object.assign({},a[l][1].end)},d={type:"strikethroughText",start:Object.assign({},a[u][1].end),end:Object.assign({},a[l][1].start)},h=[["enter",c,s],["enter",a[u][1],s],["exit",a[u][1],s],["enter",d,s]],m=s.parser.constructs.insideSpan.null;m&&mt(h,h.length,0,ls(m,a.slice(u+1,l),s)),mt(h,h.length,0,[["exit",d,s],["enter",a[l][1],s],["exit",a[l][1],s],["exit",c,s]]),mt(a,u-1,l-u+3,h),l=u+h.length-2;break}}for(l=-1;++l<a.length;)a[l][1].type==="strikethroughSequenceTemporary"&&(a[l][1].type="data");return a}function o(a,s,l){const u=this.previous,c=this.events;let d=0;return h;function h(f){return u===126&&c[c.length-1][1].type!=="characterEscape"?l(f):(a.enter("strikethroughSequenceTemporary"),m(f))}function m(f){const x=Kr(u);if(f===126)return d>1?l(f):(a.consume(f),d++,m);if(d<2&&!n)return l(f);const S=a.exit("strikethroughSequenceTemporary"),g=Kr(f);return S._open=!g||g===2&&!!x,S._close=!x||x===2&&!!g,s(f)}}}class u4{constructor(){this.map=[]}add(t,n,r){d4(this,t,n,r)}consume(t){if(this.map.sort(function(o,a){return o[0]-a[0]}),this.map.length===0)return;let n=this.map.length;const r=[];for(;n>0;)n-=1,r.push(t.slice(this.map[n][0]+this.map[n][1]),this.map[n][2]),t.length=this.map[n][0];r.push(t.slice()),t.length=0;let i=r.pop();for(;i;){for(const o of i)t.push(o);i=r.pop()}this.map.length=0}}function d4(e,t,n,r){let i=0;if(!(n===0&&r.length===0)){for(;i<e.map.length;){if(e.map[i][0]===t){e.map[i][1]+=n,e.map[i][2].push(...r);return}i+=1}e.map.push([t,n,r])}}function c4(e,t){let n=!1;const r=[];for(;t<e.length;){const i=e[t];if(n){if(i[0]==="enter")i[1].type==="tableContent"&&r.push(e[t+1][1].type==="tableDelimiterMarker"?"left":"none");else if(i[1].type==="tableContent"){if(e[t-1][1].type==="tableDelimiterMarker"){const o=r.length-1;r[o]=r[o]==="left"?"center":"right"}}else if(i[1].type==="tableDelimiterRow")break}else i[0]==="enter"&&i[1].type==="tableDelimiterRow"&&(n=!0);t+=1}return r}function p4(){return{flow:{null:{name:"table",tokenize:h4,resolveAll:m4}}}}function h4(e,t,n){const r=this;let i=0,o=0,a;return s;function s(T){let D=r.events.length-1;for(;D>-1;){const Z=r.events[D][1].type;if(Z==="lineEnding"||Z==="linePrefix")D--;else break}const W=D>-1?r.events[D][1].type:null,$=W==="tableHead"||W==="tableRow"?k:l;return $===k&&r.parser.lazy[r.now().line]?n(T):$(T)}function l(T){return e.enter("tableHead"),e.enter("tableRow"),u(T)}function u(T){return T===124||(a=!0,o+=1),c(T)}function c(T){return T===null?n(T):K(T)?o>1?(o=0,r.interrupt=!0,e.exit("tableRow"),e.enter("lineEnding"),e.consume(T),e.exit("lineEnding"),m):n(T):ne(T)?oe(e,c,"whitespace")(T):(o+=1,a&&(a=!1,i+=1),T===124?(e.enter("tableCellDivider"),e.consume(T),e.exit("tableCellDivider"),a=!0,c):(e.enter("data"),d(T)))}function d(T){return T===null||T===124||pe(T)?(e.exit("data"),c(T)):(e.consume(T),T===92?h:d)}function h(T){return T===92||T===124?(e.consume(T),d):d(T)}function m(T){return r.interrupt=!1,r.parser.lazy[r.now().line]?n(T):(e.enter("tableDelimiterRow"),a=!1,ne(T)?oe(e,f,"linePrefix",r.parser.constructs.disable.null.includes("codeIndented")?void 0:4)(T):f(T))}function f(T){return T===45||T===58?S(T):T===124?(a=!0,e.enter("tableCellDivider"),e.consume(T),e.exit("tableCellDivider"),x):z(T)}function x(T){return ne(T)?oe(e,S,"whitespace")(T):S(T)}function S(T){return T===58?(o+=1,a=!0,e.enter("tableDelimiterMarker"),e.consume(T),e.exit("tableDelimiterMarker"),g):T===45?(o+=1,g(T)):T===null||K(T)?w(T):z(T)}function g(T){return T===45?(e.enter("tableDelimiterFiller"),y(T)):z(T)}function y(T){return T===45?(e.consume(T),y):T===58?(a=!0,e.exit("tableDelimiterFiller"),e.enter("tableDelimiterMarker"),e.consume(T),e.exit("tableDelimiterMarker"),_):(e.exit("tableDelimiterFiller"),_(T))}function _(T){return ne(T)?oe(e,w,"whitespace")(T):w(T)}function w(T){return T===124?f(T):T===null||K(T)?!a||i!==o?z(T):(e.exit("tableDelimiterRow"),e.exit("tableHead"),t(T)):z(T)}function z(T){return n(T)}function k(T){return e.enter("tableRow"),E(T)}function E(T){return T===124?(e.enter("tableCellDivider"),e.consume(T),e.exit("tableCellDivider"),E):T===null||K(T)?(e.exit("tableRow"),t(T)):ne(T)?oe(e,E,"whitespace")(T):(e.enter("data"),N(T))}function N(T){return T===null||T===124||pe(T)?(e.exit("data"),E(T)):(e.consume(T),T===92?q:N)}function q(T){return T===92||T===124?(e.consume(T),N):N(T)}}function m4(e,t){let n=-1,r=!0,i=0,o=[0,0,0,0],a=[0,0,0,0],s=!1,l=0,u,c,d;const h=new u4;for(;++n<e.length;){const m=e[n],f=m[1];m[0]==="enter"?f.type==="tableHead"?(s=!1,l!==0&&(ch(h,t,l,u,c),c=void 0,l=0),u={type:"table",start:Object.assign({},f.start),end:Object.assign({},f.end)},h.add(n,0,[["enter",u,t]])):f.type==="tableRow"||f.type==="tableDelimiterRow"?(r=!0,d=void 0,o=[0,0,0,0],a=[0,n+1,0,0],s&&(s=!1,c={type:"tableBody",start:Object.assign({},f.start),end:Object.assign({},f.end)},h.add(n,0,[["enter",c,t]])),i=f.type==="tableDelimiterRow"?2:c?3:1):i&&(f.type==="data"||f.type==="tableDelimiterMarker"||f.type==="tableDelimiterFiller")?(r=!1,a[2]===0&&(o[1]!==0&&(a[0]=a[1],d=qo(h,t,o,i,void 0,d),o=[0,0,0,0]),a[2]=n)):f.type==="tableCellDivider"&&(r?r=!1:(o[1]!==0&&(a[0]=a[1],d=qo(h,t,o,i,void 0,d)),o=a,a=[o[1],n,0,0])):f.type==="tableHead"?(s=!0,l=n):f.type==="tableRow"||f.type==="tableDelimiterRow"?(l=n,o[1]!==0?(a[0]=a[1],d=qo(h,t,o,i,n,d)):a[1]!==0&&(d=qo(h,t,a,i,n,d)),i=0):i&&(f.type==="data"||f.type==="tableDelimiterMarker"||f.type==="tableDelimiterFiller")&&(a[3]=n)}for(l!==0&&ch(h,t,l,u,c),h.consume(t.events),n=-1;++n<t.events.length;){const m=t.events[n];m[0]==="enter"&&m[1].type==="table"&&(m[1]._align=c4(t.events,n))}return e}function qo(e,t,n,r,i,o){const a=r===1?"tableHeader":r===2?"tableDelimiter":"tableData",s="tableContent";n[0]!==0&&(o.end=Object.assign({},gr(t.events,n[0])),e.add(n[0],0,[["exit",o,t]]));const l=gr(t.events,n[1]);if(o={type:a,start:Object.assign({},l),end:Object.assign({},l)},e.add(n[1],0,[["enter",o,t]]),n[2]!==0){const u=gr(t.events,n[2]),c=gr(t.events,n[3]),d={type:s,start:Object.assign({},u),end:Object.assign({},c)};if(e.add(n[2],0,[["enter",d,t]]),r!==2){const h=t.events[n[2]],m=t.events[n[3]];if(h[1].end=Object.assign({},m[1].end),h[1].type="chunkText",h[1].contentType="text",n[3]>n[2]+1){const f=n[2]+1,x=n[3]-n[2]-1;e.add(f,x,[])}}e.add(n[3]+1,0,[["exit",d,t]])}return i!==void 0&&(o.end=Object.assign({},gr(t.events,i)),e.add(i,0,[["exit",o,t]]),o=void 0),o}function ch(e,t,n,r,i){const o=[],a=gr(t.events,n);i&&(i.end=Object.assign({},a),o.push(["exit",i,t])),r.end=Object.assign({},a),o.push(["exit",r,t]),e.add(n+1,0,o)}function gr(e,t){const n=e[t],r=n[0]==="enter"?"start":"end";return n[1][r]}const f4={name:"tasklistCheck",tokenize:_4};function g4(){return{text:{91:f4}}}function _4(e,t,n){const r=this;return i;function i(l){return r.previous!==null||!r._gfmTasklistFirstContentOfListItem?n(l):(e.enter("taskListCheck"),e.enter("taskListCheckMarker"),e.consume(l),e.exit("taskListCheckMarker"),o)}function o(l){return pe(l)?(e.enter("taskListCheckValueUnchecked"),e.consume(l),e.exit("taskListCheckValueUnchecked"),a):l===88||l===120?(e.enter("taskListCheckValueChecked"),e.consume(l),e.exit("taskListCheckValueChecked"),a):n(l)}function a(l){return l===93?(e.enter("taskListCheckMarker"),e.consume(l),e.exit("taskListCheckMarker"),e.exit("taskListCheck"),s):n(l)}function s(l){return K(l)?t(l):ne(l)?e.check({tokenize:y4},t,n)(l):n(l)}}function y4(e,t,n){return oe(e,r,"whitespace");function r(i){return i===null?n(i):t(i)}}function x4(e){return v0([UC(),e4(),l4(e),p4(),g4()])}const b4={};function hg(e){const t=this,n=e||b4,r=t.data(),i=r.micromarkExtensions||(r.micromarkExtensions=[]),o=r.fromMarkdownExtensions||(r.fromMarkdownExtensions=[]),a=r.toMarkdownExtensions||(r.toMarkdownExtensions=[]);i.push(x4(n)),o.push(OC()),a.push(WC(n))}function v4(){const{sectionId:e}=Xf(),{getProblemProgress:t,getSectionProgress:n}=lo(),r=Cn.find(u=>u.id===e),i=e?bb(e):[],[o,a]=C.useState(()=>e?localStorage.getItem(`section-intro-collapsed-${e}`)==="true":!1);C.useEffect(()=>{if(e){const u=localStorage.getItem(`section-intro-collapsed-${e}`);a(u==="true")}},[e]);const s=()=>{const u=!o;a(u),e&&localStorage.setItem(`section-intro-collapsed-${e}`,String(u))};if(e&&gp[e])return p.jsx(Gf,{replace:!0,to:`/section/${gp[e]}`});if(!r)return p.jsxs("div",{className:"text-center py-12",children:[p.jsx("h1",{className:"text-2xl font-bold text-gray-900 dark:text-dark-100 mb-4",children:"Section Not Found"}),p.jsx(Ye,{to:"/",className:"text-primary-500 hover:text-primary-400",children:"Return to Home"})]});const l=n(r.id,r.problems.length);return p.jsxs("div",{className:"max-w-4xl mx-auto",children:[p.jsx(ei,{title:r.title,description:`${r.description} Learn ${r.title.toLowerCase()} with ${r.problems.length} hands-on coding problems.`,canonical:`/section/${r.id}`}),p.jsxs("div",{className:"mb-8",children:[p.jsxs("div",{className:"flex items-center gap-4 mb-4",children:[p.jsx("div",{className:"w-14 h-14 bg-white dark:bg-dark-600 border border-gray-200 dark:border-dark-500 rounded-xl flex items-center justify-center text-3xl",children:r.icon}),p.jsxs("div",{children:[p.jsx("h1",{className:"text-2xl font-bold text-gray-900 dark:text-dark-100 tracking-tight",children:r.title}),p.jsx("p",{className:"text-gray-500 dark:text-dark-200",children:r.description})]})]}),p.jsxs("div",{className:"flex items-center gap-4",children:[p.jsx("div",{className:"flex-1 h-1.5 bg-gray-200 dark:bg-dark-500 rounded-full overflow-hidden",children:p.jsx("div",{className:"h-full progress-gradient transition-all duration-300 rounded-full",style:{width:`${l}%`}})}),p.jsxs("span",{className:"text-primary-600 dark:text-primary-300 font-medium font-mono text-sm",children:[l,"%"]})]})]}),p.jsxs("div",{className:"bg-white dark:bg-dark-800 rounded-xl p-6 mb-8 border border-gray-200 dark:border-dark-500",children:[p.jsxs("button",{onClick:s,className:"w-full flex items-center justify-between text-left",children:[p.jsx("h2",{className:"text-lg font-semibold text-gray-900 dark:text-dark-100",children:"Introduction"}),p.jsx("svg",{className:`w-5 h-5 text-gray-500 dark:text-dark-200 transition-transform duration-200 ${o?"":"rotate-180"}`,fill:"none",stroke:"currentColor",viewBox:"0 0 24 24",children:p.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:2,d:"M19 9l-7 7-7-7"})})]}),!o&&p.jsx("div",{className:"prose prose-sm max-w-none mt-4",children:p.jsx(B0,{remarkPlugins:[hg],components:{h1:({children:u})=>p.jsx("h1",{className:"text-xl font-bold text-gray-900 dark:text-dark-100 mt-0 mb-4",children:u}),h2:({children:u})=>p.jsx("h2",{className:"text-lg font-semibold text-gray-900 dark:text-dark-100 mt-6 mb-3",children:u}),h3:({children:u})=>p.jsx("h3",{className:"text-base font-medium text-gray-700 dark:text-dark-200 mt-4 mb-2",children:u}),p:({children:u})=>p.jsx("p",{className:"text-gray-600 dark:text-dark-200 mb-3 leading-relaxed",children:u}),pre:({children:u})=>p.jsx("pre",{className:"bg-gray-100 dark:bg-dark-900 rounded-lg p-4 overflow-x-auto my-3 text-sm text-gray-800 dark:text-dark-100",children:u}),code:({className:u,children:c})=>!u&&!String(c).includes(`
`)?p.jsx("code",{className:"bg-gray-100 dark:bg-dark-600 px-1.5 py-0.5 rounded text-primary-600 dark:text-primary-300 text-sm",children:c}):p.jsx("code",{className:"text-sm",children:c}),ul:({children:u})=>p.jsx("ul",{className:"list-disc list-inside text-gray-600 dark:text-dark-200 space-y-1 mb-3",children:u}),li:({children:u})=>p.jsx("li",{className:"text-gray-600 dark:text-dark-200",children:u}),strong:({children:u})=>p.jsx("strong",{className:"text-gray-900 dark:text-dark-100 font-semibold",children:u}),table:({children:u})=>p.jsx("div",{className:"overflow-x-auto my-4",children:p.jsx("table",{className:"min-w-full border-collapse border border-gray-300 dark:border-dark-500 text-sm",children:u})}),thead:({children:u})=>p.jsx("thead",{className:"bg-gray-100 dark:bg-dark-600",children:u}),tbody:({children:u})=>p.jsx("tbody",{className:"divide-y divide-gray-200 dark:divide-dark-500",children:u}),tr:({children:u})=>p.jsx("tr",{children:u}),th:({children:u})=>p.jsx("th",{className:"border border-gray-300 dark:border-dark-500 px-3 py-2 text-left font-semibold text-gray-700 dark:text-dark-200",children:u}),td:({children:u})=>p.jsx("td",{className:"border border-gray-300 dark:border-dark-500 px-3 py-2 text-gray-600 dark:text-dark-200",children:u})},children:r.introduction})})]}),p.jsxs("div",{children:[p.jsx("h2",{className:"text-lg font-semibold text-gray-900 dark:text-dark-100 mb-4",children:"Problems"}),p.jsx("div",{className:"space-y-3",children:i.map((u,c)=>{const d=t(r.id,u.id);return p.jsxs(Ye,{to:`/problem/${r.id}/${u.id}`,className:"flex items-center gap-4 bg-white dark:bg-dark-800 rounded-lg p-4 border border-gray-200 dark:border-dark-500 hover:border-primary-400 dark:hover:border-primary-500/30 hover:shadow-sm transition-all group",children:[p.jsx("div",{className:`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${d.status==="completed"?"bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400":d.status==="in_progress"?"bg-yellow-100 text-yellow-600 dark:bg-yellow-900/30 dark:text-yellow-400":"bg-gray-100 text-gray-500 dark:bg-dark-600 dark:text-dark-200"}`,children:d.status==="completed"?"✓":c+1}),p.jsxs("div",{className:"flex-1",children:[p.jsx("h3",{className:"text-gray-900 dark:text-dark-100 font-medium group-hover:text-primary-600 dark:group-hover:text-primary-300 transition-colors",children:u.title}),p.jsxs("div",{className:"flex items-center gap-3 mt-1",children:[p.jsx("span",{className:`text-xs px-2 py-0.5 rounded ${u.difficulty==="easy"?"bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400":u.difficulty==="medium"?"bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400":"bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"}`,children:u.difficulty}),p.jsxs("span",{className:"text-gray-400 dark:text-dark-300 text-xs font-mono",children:[u.testCases.length," tests"]}),u.pytorch&&p.jsxs("span",{className:`text-xs font-mono ${d.solvedIn.includes("pytorch")?"text-green-600 dark:text-green-400":"text-gray-400 dark:text-dark-300"}`,title:d.solvedIn.includes("pytorch")?"Solved in PyTorch":"PyTorch track available",children:[d.solvedIn.includes("pytorch")?"✓ ":"","PyTorch"]}),d.solutionViewed&&p.jsx("span",{className:"text-xs text-gray-400 dark:text-dark-300",title:"You revealed the reference solution for this problem",children:"solution viewed"})]})]}),p.jsx("div",{className:"text-gray-400 dark:text-dark-300 group-hover:text-primary-500 dark:group-hover:text-primary-300 transition-colors",children:"→"})]},u.id)})})]})]})}var mg={exports:{}},k4="SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED",w4=k4,T4=w4;function fg(){}function gg(){}gg.resetWarningCache=fg;var C4=function(){function e(r,i,o,a,s,l){if(l!==T4){var u=new Error("Calling PropTypes validators directly is not supported by the `prop-types` package. Use PropTypes.checkPropTypes() to call them. Read more at http://fb.me/use-check-prop-types");throw u.name="Invariant Violation",u}}e.isRequired=e;function t(){return e}var n={array:e,bigint:e,bool:e,func:e,number:e,object:e,string:e,symbol:e,any:e,arrayOf:t,element:e,elementType:e,instanceOf:t,node:e,objectOf:t,oneOf:t,oneOfType:t,shape:t,exact:t,checkPropTypes:gg,resetWarningCache:fg};return n.PropTypes=n,n};mg.exports=C4();var S4=mg.exports;const ue=ar(S4);var xt=typeof window<"u"?window:null,Ud=xt===null,to=Ud?void 0:xt.document,Nt="addEventListener",At="removeEventListener",al="getBoundingClientRect",fi="_a",Lt="_b",Zt="_c",Oo="horizontal",jt=function(){return!1},z4=Ud?"calc":["","-webkit-","-moz-","-o-"].filter(function(e){var t=to.createElement("div");return t.style.cssText="width:"+e+"calc(9px)",!!t.style.length}).shift()+"calc",_g=function(e){return typeof e=="string"||e instanceof String},ph=function(e){if(_g(e)){var t=to.querySelector(e);if(!t)throw new Error("Selector "+e+" did not match a DOM element");return t}return e},Ve=function(e,t,n){var r=e[t];return r!==void 0?r:n},Wo=function(e,t,n,r){if(t){if(r==="end")return 0;if(r==="center")return e/2}else if(n){if(r==="start")return 0;if(r==="center")return e/2}return e},E4=function(e,t){var n=to.createElement("div");return n.className="gutter gutter-"+t,n},P4=function(e,t,n){var r={};return _g(t)?r[e]=t:r[e]=z4+"("+t+"% - "+n+"px)",r},N4=function(e,t){var n;return n={},n[e]=t+"px",n},hh=function(e,t){if(t===void 0&&(t={}),Ud)return{};var n=e,r,i,o,a,s,l;Array.from&&(n=Array.from(n));var u=ph(n[0]),c=u.parentNode,d=getComputedStyle?getComputedStyle(c):null,h=d?d.flexDirection:null,m=Ve(t,"sizes")||n.map(function(){return 100/n.length}),f=Ve(t,"minSize",100),x=Array.isArray(f)?f:n.map(function(){return f}),S=Ve(t,"maxSize",1/0),g=Array.isArray(S)?S:n.map(function(){return S}),y=Ve(t,"expandToMin",!1),_=Ve(t,"gutterSize",10),w=Ve(t,"gutterAlign","center"),z=Ve(t,"snapOffset",30),k=Array.isArray(z)?z:n.map(function(){return z}),E=Ve(t,"dragInterval",1),N=Ve(t,"direction",Oo),q=Ve(t,"cursor",N===Oo?"col-resize":"row-resize"),T=Ve(t,"gutter",E4),D=Ve(t,"elementStyle",P4),W=Ve(t,"gutterStyle",N4);N===Oo?(r="width",i="clientX",o="left",a="right",s="clientWidth"):N==="vertical"&&(r="height",i="clientY",o="top",a="bottom",s="clientHeight");function $(O,A,I,Q){var xe=D(r,A,I,Q);Object.keys(xe).forEach(function(fe){O.style[fe]=xe[fe]})}function Z(O,A,I){var Q=W(r,A,I);Object.keys(Q).forEach(function(xe){O.style[xe]=Q[xe]})}function F(){return l.map(function(O){return O.size})}function X(O){return"touches"in O?O.touches[0][i]:O[i]}function ie(O){var A=l[this.a],I=l[this.b],Q=A.size+I.size;A.size=O/this.size*Q,I.size=Q-O/this.size*Q,$(A.element,A.size,this[Lt],A.i),$(I.element,I.size,this[Zt],I.i)}function R(O){var A,I=l[this.a],Q=l[this.b];this.dragging&&(A=X(O)-this.start+(this[Lt]-this.dragOffset),E>1&&(A=Math.round(A/E)*E),A<=I.minSize+I.snapOffset+this[Lt]?A=I.minSize+this[Lt]:A>=this.size-(Q.minSize+Q.snapOffset+this[Zt])&&(A=this.size-(Q.minSize+this[Zt])),A>=I.maxSize-I.snapOffset+this[Lt]?A=I.maxSize+this[Lt]:A<=this.size-(Q.maxSize-Q.snapOffset+this[Zt])&&(A=this.size-(Q.maxSize+this[Zt])),ie.call(this,A),Ve(t,"onDrag",jt)(F()))}function U(){var O=l[this.a].element,A=l[this.b].element,I=O[al](),Q=A[al]();this.size=I[r]+Q[r]+this[Lt]+this[Zt],this.start=I[o],this.end=I[a]}function b(O){if(!getComputedStyle)return null;var A=getComputedStyle(O);if(!A)return null;var I=O[s];return I===0?null:(N===Oo?I-=parseFloat(A.paddingLeft)+parseFloat(A.paddingRight):I-=parseFloat(A.paddingTop)+parseFloat(A.paddingBottom),I)}function V(O){var A=b(c);if(A===null||x.reduce(function(fe,Ae){return fe+Ae},0)>A)return O;var I=0,Q=[],xe=O.map(function(fe,Ae){var yt=A*fe/100,$t=Wo(_,Ae===0,Ae===O.length-1,w),Et=x[Ae]+$t;return yt<Et?(I+=Et-yt,Q.push(0),Et):(Q.push(yt-Et),yt)});return I===0?O:xe.map(function(fe,Ae){var yt=fe;if(I>0&&Q[Ae]-I>0){var $t=Math.min(I,Q[Ae]-I);I-=$t,yt=fe-$t}return yt/A*100})}function H(){var O=this,A=l[O.a].element,I=l[O.b].element;O.dragging&&Ve(t,"onDragEnd",jt)(F()),O.dragging=!1,xt[At]("mouseup",O.stop),xt[At]("touchend",O.stop),xt[At]("touchcancel",O.stop),xt[At]("mousemove",O.move),xt[At]("touchmove",O.move),O.stop=null,O.move=null,A[At]("selectstart",jt),A[At]("dragstart",jt),I[At]("selectstart",jt),I[At]("dragstart",jt),A.style.userSelect="",A.style.webkitUserSelect="",A.style.MozUserSelect="",A.style.pointerEvents="",I.style.userSelect="",I.style.webkitUserSelect="",I.style.MozUserSelect="",I.style.pointerEvents="",O.gutter.style.cursor="",O.parent.style.cursor="",to.body.style.cursor=""}function v(O){if(!("button"in O&&O.button!==0)){var A=this,I=l[A.a].element,Q=l[A.b].element;A.dragging||Ve(t,"onDragStart",jt)(F()),O.preventDefault(),A.dragging=!0,A.move=R.bind(A),A.stop=H.bind(A),xt[Nt]("mouseup",A.stop),xt[Nt]("touchend",A.stop),xt[Nt]("touchcancel",A.stop),xt[Nt]("mousemove",A.move),xt[Nt]("touchmove",A.move),I[Nt]("selectstart",jt),I[Nt]("dragstart",jt),Q[Nt]("selectstart",jt),Q[Nt]("dragstart",jt),I.style.userSelect="none",I.style.webkitUserSelect="none",I.style.MozUserSelect="none",I.style.pointerEvents="none",Q.style.userSelect="none",Q.style.webkitUserSelect="none",Q.style.MozUserSelect="none",Q.style.pointerEvents="none",A.gutter.style.cursor=q,A.parent.style.cursor=q,to.body.style.cursor=q,U.call(A),A.dragOffset=X(O)-A.end}}m=V(m);var te=[];l=n.map(function(O,A){var I={element:ph(O),size:m[A],minSize:x[A],maxSize:g[A],snapOffset:k[A],i:A},Q;if(A>0&&(Q={a:A-1,b:A,dragging:!1,direction:N,parent:c},Q[Lt]=Wo(_,A-1===0,!1,w),Q[Zt]=Wo(_,!1,A===n.length-1,w),h==="row-reverse"||h==="column-reverse")){var xe=Q.a;Q.a=Q.b,Q.b=xe}if(A>0){var fe=T(A,N,I.element);Z(fe,_,A),Q[fi]=v.bind(Q),fe[Nt]("mousedown",Q[fi]),fe[Nt]("touchstart",Q[fi]),c.insertBefore(fe,I.element),Q.gutter=fe}return $(I.element,I.size,Wo(_,A===0,A===n.length-1,w),A),A>0&&te.push(Q),I});function ce(O){var A=O.i===te.length,I=A?te[O.i-1]:te[O.i];U.call(I);var Q=A?I.size-O.minSize-I[Zt]:O.minSize+I[Lt];ie.call(I,Q)}l.forEach(function(O){var A=O.element[al]()[r];A<O.minSize&&(y?ce(O):O.minSize=A)});function se(O){var A=V(O);A.forEach(function(I,Q){if(Q>0){var xe=te[Q-1],fe=l[xe.a],Ae=l[xe.b];fe.size=A[Q-1],Ae.size=I,$(fe.element,fe.size,xe[Lt],fe.i),$(Ae.element,Ae.size,xe[Zt],Ae.i)}})}function et(O,A){te.forEach(function(I){if(A!==!0?I.parent.removeChild(I.gutter):(I.gutter[At]("mousedown",I[fi]),I.gutter[At]("touchstart",I[fi])),O!==!0){var Q=D(r,I.a.size,I[Lt]);Object.keys(Q).forEach(function(xe){l[I.a].element.style[xe]="",l[I.b].element.style[xe]=""})}})}return{setSizes:se,getSizes:F,collapse:function(A){ce(l[A])},destroy:et,parent:c,pairs:te}};function sl(e,t){var n={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&t.indexOf(r)===-1&&(n[r]=e[r]);return n}var qr=function(e){function t(){e.apply(this,arguments)}return e&&(t.__proto__=e),t.prototype=Object.create(e&&e.prototype),t.prototype.constructor=t,t.prototype.componentDidMount=function(){var r=this.props;r.children;var i=r.gutter,o=sl(r,["children","gutter"]),a=o;a.gutter=function(s,l){var u;return i?u=i(s,l):(u=document.createElement("div"),u.className="gutter gutter-"+l),u.__isSplitGutter=!0,u},this.split=hh(this.parent.children,a)},t.prototype.componentDidUpdate=function(r){var i=this,o=this.props;o.children;var a=o.minSize,s=o.sizes,l=o.collapsed,u=sl(o,["children","minSize","sizes","collapsed"]),c=u,d=r.minSize,h=r.sizes,m=r.collapsed,f=["maxSize","expandToMin","gutterSize","gutterAlign","snapOffset","dragInterval","direction","cursor"],x=f.map(function(y){return i.props[y]!==r[y]}).reduce(function(y,_){return y||_},!1);if(Array.isArray(a)&&Array.isArray(d)){var S=!1;a.forEach(function(y,_){S=S||y!==d[_]}),x=x||S}else Array.isArray(a)||Array.isArray(d)?x=!0:x=x||a!==d;if(x)c.minSize=a,c.sizes=s||this.split.getSizes(),this.split.destroy(!0,!0),c.gutter=function(y,_,w){return w.previousSibling},this.split=hh(Array.from(this.parent.children).filter(function(y){return!y.__isSplitGutter}),c);else if(s){var g=!1;s.forEach(function(y,_){g=g||y!==h[_]}),g&&this.split.setSizes(this.props.sizes)}Number.isInteger(l)&&(l!==m||x)&&this.split.collapse(l)},t.prototype.componentWillUnmount=function(){this.split.destroy(),delete this.split},t.prototype.render=function(){var r=this,i=this.props;i.sizes,i.minSize,i.maxSize,i.expandToMin,i.gutterSize,i.gutterAlign,i.snapOffset,i.dragInterval,i.direction,i.cursor,i.gutter,i.elementStyle,i.gutterStyle,i.onDrag,i.onDragStart,i.onDragEnd,i.collapsed;var o=i.children,a=sl(i,["sizes","minSize","maxSize","expandToMin","gutterSize","gutterAlign","snapOffset","dragInterval","direction","cursor","gutter","elementStyle","gutterStyle","onDrag","onDragStart","onDragEnd","collapsed","children"]),s=a;return Re.createElement("div",Object.assign({},{ref:function(l){r.parent=l}},s),o)},t}(Re.Component);qr.propTypes={sizes:ue.arrayOf(ue.number),minSize:ue.oneOfType([ue.number,ue.arrayOf(ue.number)]),maxSize:ue.oneOfType([ue.number,ue.arrayOf(ue.number)]),expandToMin:ue.bool,gutterSize:ue.number,gutterAlign:ue.string,snapOffset:ue.oneOfType([ue.number,ue.arrayOf(ue.number)]),dragInterval:ue.number,direction:ue.string,cursor:ue.string,gutter:ue.func,elementStyle:ue.func,gutterStyle:ue.func,onDrag:ue.func,onDragStart:ue.func,onDragEnd:ue.func,collapsed:ue.number,children:ue.arrayOf(ue.element)};qr.defaultProps={sizes:void 0,minSize:void 0,maxSize:void 0,expandToMin:void 0,gutterSize:void 0,gutterAlign:void 0,snapOffset:void 0,dragInterval:void 0,direction:void 0,cursor:void 0,gutter:void 0,elementStyle:void 0,gutterStyle:void 0,onDrag:void 0,onDragStart:void 0,onDragEnd:void 0,collapsed:void 0,children:void 0};const Cu=3e4;class A4{constructor(){Ie(this,"worker",null);Ie(this,"ready",null);Ie(this,"status","idle");Ie(this,"initError",null);Ie(this,"listeners",new Set);Ie(this,"pending",null);Ie(this,"queue",Promise.resolve());Ie(this,"nextId",1)}getStatus(){return this.status}getInitError(){return this.initError}subscribe(t){return this.listeners.add(t),()=>this.listeners.delete(t)}setStatus(t,n=null){this.status=t,this.initError=n,this.listeners.forEach(r=>r())}start(){if(this.ready)return this.ready;if(typeof Worker>"u")return this.setStatus("error","Web Workers are not supported in this browser"),Promise.reject(new Error(this.initError));const t="/ml-coding-lab/",n=new Worker(`${t}pyodide-worker.js`);return this.worker=n,this.setStatus("loading"),this.ready=new Promise((i,o)=>{n.onmessage=a=>{const s=a.data;s.type==="ready"?(this.setStatus("ready"),i()):s.type==="init-error"?(this.setStatus("error",s.error),this.ready=null,o(new Error(s.error))):this.handleMessage(s)},n.onerror=a=>{const s=a.message||"Python worker crashed";this.status==="loading"&&(this.setStatus("error",s),this.ready=null,o(new Error(s)))}}),n.postMessage({type:"init",version:"mw8cdbgp"}),this.ready.catch(()=>{}),this.ready}handleMessage(t){const n=this.pending;!n||t.id!==n.id||(t.type==="stdout"||t.type==="stderr"?n.onOutput(t.text??"",t.type):t.type==="done"?this.finish(n,t.error?{error:t.error}:{}):t.type==="test-results"&&this.finish(n,t.error?{error:t.error}:{results:t.results}))}finish(t,n){clearTimeout(t.timer),this.pending===t&&(this.pending=null),t.resolve(n)}restart(t){var r;const n=this.pending;(r=this.worker)==null||r.terminate(),this.worker=null,this.ready=null,this.setStatus("idle"),n&&this.finish(n,{aborted:t}),this.start().catch(()=>{})}stop(){this.pending&&this.restart("stopped")}enqueue(t,n,r){const i=async()=>{try{await this.start()}catch(a){return{error:a instanceof Error?a.message:"Python failed to load"}}return new Promise(a=>{const s=this.nextId++,l={id:s,onOutput:n,resolve:a,timer:setTimeout(()=>this.restart("timeout"),r)};this.pending=l,this.worker.postMessage({...t,id:s})})},o=this.queue.then(i,i);return this.queue=o.catch(()=>{}),o}runCode(t,n,r=Cu){return this.enqueue({type:"run",code:t},n,r)}runTests(t,n,r,i,o,a=Cu){const s=n.map(({id:l,input:u,expected:c})=>({id:l,input:u,expected:c}));return this.enqueue({type:"test",code:t,testCases:s,functionName:r,language:i},o,a)}}const _n=new A4;function mh(e){return e.aborted==="timeout"?`Stopped: your code ran longer than ${Cu/1e3}s (infinite loop?). Python is restarting…`:e.aborted==="stopped"?"Stopped. Python is restarting…":null}function yg(){const e=C.useSyncExternalStore(l=>_n.subscribe(l),()=>_n.getStatus()),[t,n]=C.useState([]);C.useEffect(()=>{_n.start().catch(()=>{})},[]),C.useEffect(()=>{e==="ready"?n(l=>l.length===0?["Python environment ready with NumPy loaded."]:l):e==="error"&&n(l=>[...l,`[Error] ${_n.getInitError()??"Failed to load Python"}`])},[e]);const r=C.useCallback((l,u="")=>{const c=l.split(`
`).filter(d=>d.trim());c.length&&n(d=>[...d,...c.map(h=>u+h)])},[]),i=C.useCallback(()=>n([]),[]),o=C.useCallback(async l=>{n([]);const u=await _n.runCode(l,(d,h)=>r(d,h==="stderr"?"[Error] ":"")),c=mh(u);return c?(r(c,"[Error] "),{success:!1,error:c}):u.error?(r(u.error,"[Error] "),{success:!1,error:u.error}):{success:!0}},[r]),a=C.useCallback(async(l,u,c,d="numpy",h={})=>{const m=h.silent??!1;m||n([]);const f=await _n.runTests(l,u,c,d,m?()=>{}:(_,w)=>r(_,w==="stderr"?"[Error] ":"")),x=_=>u.map(w=>({id:w.id,passed:!1,description:w.description,expected:w.expected,actual:_,hidden:w.hidden})),S=mh(f);if(S||f.error||!f.results){const _=S??f.error??"Test execution error";return m||r(_,"[Error] "),x(_)}const g=new Map(f.results.map(_=>[_.id,_])),y=u.map(_=>{const w=g.get(_.id);return{id:_.id,passed:(w==null?void 0:w.passed)??!1,description:_.description,expected:_.expected,actual:(w==null?void 0:w.actual)??"Not run",hidden:_.hidden}});if(!m)for(const _ of y){const w=_.hidden?`Test ${_.id} (hidden)`:`Test ${_.id}`;_.passed?r(`${w}: PASSED`):_.hidden?r(`${w}: FAILED`):r(`${w}: FAILED - Expected ${_.expected}, got ${_.actual}`)}return y},[r]),s=C.useCallback(()=>_n.stop(),[]);return{isLoading:e==="loading"||e==="idle",isReady:e==="ready",error:e==="error"?_n.getInitError():null,output:t,runCode:o,runTests:a,stop:s,clearOutput:i}}function fh(e,t){(t==null||t>e.length)&&(t=e.length);for(var n=0,r=Array(t);n<t;n++)r[n]=e[n];return r}function L4(e){if(Array.isArray(e))return e}function j4(e,t,n){return(t=O4(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function R4(e,t){var n=e==null?null:typeof Symbol<"u"&&e[Symbol.iterator]||e["@@iterator"];if(n!=null){var r,i,o,a,s=[],l=!0,u=!1;try{if(o=(n=n.call(e)).next,t!==0)for(;!(l=(r=o.call(n)).done)&&(s.push(r.value),s.length!==t);l=!0);}catch(c){u=!0,i=c}finally{try{if(!l&&n.return!=null&&(a=n.return(),Object(a)!==a))return}finally{if(u)throw i}}return s}}function F4(){throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function gh(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(i){return Object.getOwnPropertyDescriptor(e,i).enumerable})),n.push.apply(n,r)}return n}function _h(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?gh(Object(n),!0).forEach(function(r){j4(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):gh(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function I4(e,t){if(e==null)return{};var n,r,i=D4(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)===-1&&{}.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}function D4(e,t){if(e==null)return{};var n={};for(var r in e)if({}.hasOwnProperty.call(e,r)){if(t.indexOf(r)!==-1)continue;n[r]=e[r]}return n}function M4(e,t){return L4(e)||R4(e,t)||W4(e,t)||F4()}function q4(e,t){if(typeof e!="object"||!e)return e;var n=e[Symbol.toPrimitive];if(n!==void 0){var r=n.call(e,t);if(typeof r!="object")return r;throw new TypeError("@@toPrimitive must return a primitive value.")}return(t==="string"?String:Number)(e)}function O4(e){var t=q4(e,"string");return typeof t=="symbol"?t:t+""}function W4(e,t){if(e){if(typeof e=="string")return fh(e,t);var n={}.toString.call(e).slice(8,-1);return n==="Object"&&e.constructor&&(n=e.constructor.name),n==="Map"||n==="Set"?Array.from(e):n==="Arguments"||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?fh(e,t):void 0}}function V4(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function yh(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(i){return Object.getOwnPropertyDescriptor(e,i).enumerable})),n.push.apply(n,r)}return n}function xh(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?yh(Object(n),!0).forEach(function(r){V4(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):yh(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function B4(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return function(r){return t.reduceRight(function(i,o){return o(i)},r)}}function bi(e){return function t(){for(var n=this,r=arguments.length,i=new Array(r),o=0;o<r;o++)i[o]=arguments[o];return i.length>=e.length?e.apply(this,i):function(){for(var a=arguments.length,s=new Array(a),l=0;l<a;l++)s[l]=arguments[l];return t.apply(n,[].concat(i,s))}}}function qa(e){return{}.toString.call(e).includes("Object")}function U4(e){return!Object.keys(e).length}function no(e){return typeof e=="function"}function H4(e,t){return Object.prototype.hasOwnProperty.call(e,t)}function Q4(e,t){return qa(t)||Fn("changeType"),Object.keys(t).some(function(n){return!H4(e,n)})&&Fn("changeField"),t}function X4(e){no(e)||Fn("selectorType")}function K4(e){no(e)||qa(e)||Fn("handlerType"),qa(e)&&Object.values(e).some(function(t){return!no(t)})&&Fn("handlersType")}function Y4(e){e||Fn("initialIsRequired"),qa(e)||Fn("initialType"),U4(e)&&Fn("initialContent")}function $4(e,t){throw new Error(e[t]||e.default)}var G4={initialIsRequired:"initial state is required",initialType:"initial state should be an object",initialContent:"initial state shouldn't be an empty object",handlerType:"handler should be an object or a function",handlersType:"all handlers should be a functions",selectorType:"selector should be a function",changeType:"provided value of changes should be an object",changeField:'it seams you want to change a field in the state which is not specified in the "initial" state',default:"an unknown error accured in `state-local` package"},Fn=bi($4)(G4),Vo={changes:Q4,selector:X4,handler:K4,initial:Y4};function Z4(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};Vo.initial(e),Vo.handler(t);var n={current:e},r=bi(tS)(n,t),i=bi(eS)(n),o=bi(Vo.changes)(e),a=bi(J4)(n);function s(){var u=arguments.length>0&&arguments[0]!==void 0?arguments[0]:function(c){return c};return Vo.selector(u),u(n.current)}function l(u){B4(r,i,o,a)(u)}return[s,l]}function J4(e,t){return no(t)?t(e.current):t}function eS(e,t){return e.current=xh(xh({},e.current),t),t}function tS(e,t,n){return no(t)?t(e.current):Object.keys(n).forEach(function(r){var i;return(i=t[r])===null||i===void 0?void 0:i.call(t,e.current[r])}),n}var nS={create:Z4},rS={paths:{vs:"https://cdn.jsdelivr.net/npm/monaco-editor@0.55.1/min/vs"}};function iS(e){return function t(){for(var n=this,r=arguments.length,i=new Array(r),o=0;o<r;o++)i[o]=arguments[o];return i.length>=e.length?e.apply(this,i):function(){for(var a=arguments.length,s=new Array(a),l=0;l<a;l++)s[l]=arguments[l];return t.apply(n,[].concat(i,s))}}}function oS(e){return{}.toString.call(e).includes("Object")}function aS(e){return e||bh("configIsRequired"),oS(e)||bh("configType"),e.urls?(sS(),{paths:{vs:e.urls.monacoBase}}):e}function sS(){console.warn(xg.deprecation)}function lS(e,t){throw new Error(e[t]||e.default)}var xg={configIsRequired:"the configuration object is required",configType:"the configuration object should be an object",default:"an unknown error accured in `@monaco-editor/loader` package",deprecation:`Deprecation warning!
    You are using deprecated way of configuration.

    Instead of using
      monaco.config({ urls: { monacoBase: '...' } })
    use
      monaco.config({ paths: { vs: '...' } })

    For more please check the link https://github.com/suren-atoyan/monaco-loader#config
  `},bh=iS(lS)(xg),uS={config:aS},dS=function(){for(var t=arguments.length,n=new Array(t),r=0;r<t;r++)n[r]=arguments[r];return function(i){return n.reduceRight(function(o,a){return a(o)},i)}};function bg(e,t){return Object.keys(t).forEach(function(n){t[n]instanceof Object&&e[n]&&Object.assign(t[n],bg(e[n],t[n]))}),_h(_h({},e),t)}var cS={type:"cancelation",msg:"operation is manually canceled"};function ll(e){var t=!1,n=new Promise(function(r,i){e.then(function(o){return t?i(cS):r(o)}),e.catch(i)});return n.cancel=function(){return t=!0},n}var pS=["monaco"],hS=nS.create({config:rS,isInitialized:!1,resolve:null,reject:null,monaco:null}),vg=M4(hS,2),ho=vg[0],ps=vg[1];function mS(e){var t=uS.config(e),n=t.monaco,r=I4(t,pS);ps(function(i){return{config:bg(i.config,r),monaco:n}})}function fS(){var e=ho(function(t){var n=t.monaco,r=t.isInitialized,i=t.resolve;return{monaco:n,isInitialized:r,resolve:i}});if(!e.isInitialized){if(ps({isInitialized:!0}),e.monaco)return e.resolve(e.monaco),ll(ul);if(window.monaco&&window.monaco.editor)return kg(window.monaco),e.resolve(window.monaco),ll(ul);dS(gS,yS)(xS)}return ll(ul)}function gS(e){return document.body.appendChild(e)}function _S(e){var t=document.createElement("script");return e&&(t.src=e),t}function yS(e){var t=ho(function(r){var i=r.config,o=r.reject;return{config:i,reject:o}}),n=_S("".concat(t.config.paths.vs,"/loader.js"));return n.onload=function(){return e()},n.onerror=t.reject,n}function xS(){var e=ho(function(n){var r=n.config,i=n.resolve,o=n.reject;return{config:r,resolve:i,reject:o}}),t=window.require;t.config(e.config),t(["vs/editor/editor.main"],function(n){var r=n.m||n;kg(r),e.resolve(r)},function(n){e.reject(n)})}function kg(e){ho().monaco||ps({monaco:e})}function bS(){return ho(function(e){var t=e.monaco;return t})}var ul=new Promise(function(e,t){return ps({resolve:e,reject:t})}),wg={config:mS,init:fS,__getMonacoInstance:bS},vS={wrapper:{display:"flex",position:"relative",textAlign:"initial"},fullWidth:{width:"100%"},hide:{display:"none"}},dl=vS,kS={container:{display:"flex",height:"100%",width:"100%",justifyContent:"center",alignItems:"center"}},wS=kS;function TS({children:e}){return Re.createElement("div",{style:wS.container},e)}var CS=TS,SS=CS;function zS({width:e,height:t,isEditorReady:n,loading:r,_ref:i,className:o,wrapperProps:a}){return Re.createElement("section",{style:{...dl.wrapper,width:e,height:t},...a},!n&&Re.createElement(SS,null,r),Re.createElement("div",{ref:i,style:{...dl.fullWidth,...!n&&dl.hide},className:o}))}var ES=zS,Tg=C.memo(ES);function PS(e){C.useEffect(e,[])}var Cg=PS;function NS(e,t,n=!0){let r=C.useRef(!0);C.useEffect(r.current||!n?()=>{r.current=!1}:e,t)}var ct=NS;function ji(){}function Pr(e,t,n,r){return AS(e,r)||LS(e,t,n,r)}function AS(e,t){return e.editor.getModel(Sg(e,t))}function LS(e,t,n,r){return e.editor.createModel(t,n,r?Sg(e,r):void 0)}function Sg(e,t){return e.Uri.parse(t)}function jS({original:e,modified:t,language:n,originalLanguage:r,modifiedLanguage:i,originalModelPath:o,modifiedModelPath:a,keepCurrentOriginalModel:s=!1,keepCurrentModifiedModel:l=!1,theme:u="light",loading:c="Loading...",options:d={},height:h="100%",width:m="100%",className:f,wrapperProps:x={},beforeMount:S=ji,onMount:g=ji}){let[y,_]=C.useState(!1),[w,z]=C.useState(!0),k=C.useRef(null),E=C.useRef(null),N=C.useRef(null),q=C.useRef(g),T=C.useRef(S),D=C.useRef(!1);Cg(()=>{let F=wg.init();return F.then(X=>(E.current=X)&&z(!1)).catch(X=>(X==null?void 0:X.type)!=="cancelation"&&console.error("Monaco initialization: error:",X)),()=>k.current?Z():F.cancel()}),ct(()=>{if(k.current&&E.current){let F=k.current.getOriginalEditor(),X=Pr(E.current,e||"",r||n||"text",o||"");X!==F.getModel()&&F.setModel(X)}},[o],y),ct(()=>{if(k.current&&E.current){let F=k.current.getModifiedEditor(),X=Pr(E.current,t||"",i||n||"text",a||"");X!==F.getModel()&&F.setModel(X)}},[a],y),ct(()=>{let F=k.current.getModifiedEditor();F.getOption(E.current.editor.EditorOption.readOnly)?F.setValue(t||""):t!==F.getValue()&&(F.executeEdits("",[{range:F.getModel().getFullModelRange(),text:t||"",forceMoveMarkers:!0}]),F.pushUndoStop())},[t],y),ct(()=>{var F,X;(X=(F=k.current)==null?void 0:F.getModel())==null||X.original.setValue(e||"")},[e],y),ct(()=>{let{original:F,modified:X}=k.current.getModel();E.current.editor.setModelLanguage(F,r||n||"text"),E.current.editor.setModelLanguage(X,i||n||"text")},[n,r,i],y),ct(()=>{var F;(F=E.current)==null||F.editor.setTheme(u)},[u],y),ct(()=>{var F;(F=k.current)==null||F.updateOptions(d)},[d],y);let W=C.useCallback(()=>{var ie;if(!E.current)return;T.current(E.current);let F=Pr(E.current,e||"",r||n||"text",o||""),X=Pr(E.current,t||"",i||n||"text",a||"");(ie=k.current)==null||ie.setModel({original:F,modified:X})},[n,t,i,e,r,o,a]),$=C.useCallback(()=>{var F;!D.current&&N.current&&(k.current=E.current.editor.createDiffEditor(N.current,{automaticLayout:!0,...d}),W(),(F=E.current)==null||F.editor.setTheme(u),_(!0),D.current=!0)},[d,u,W]);C.useEffect(()=>{y&&q.current(k.current,E.current)},[y]),C.useEffect(()=>{!w&&!y&&$()},[w,y,$]);function Z(){var X,ie,R,U;let F=(X=k.current)==null?void 0:X.getModel();s||((ie=F==null?void 0:F.original)==null||ie.dispose()),l||((R=F==null?void 0:F.modified)==null||R.dispose()),(U=k.current)==null||U.dispose()}return Re.createElement(Tg,{width:m,height:h,isEditorReady:y,loading:c,_ref:N,className:f,wrapperProps:x})}var RS=jS;C.memo(RS);function FS(e){let t=C.useRef();return C.useEffect(()=>{t.current=e},[e]),t.current}var IS=FS,Bo=new Map;function DS({defaultValue:e,defaultLanguage:t,defaultPath:n,value:r,language:i,path:o,theme:a="light",line:s,loading:l="Loading...",options:u={},overrideServices:c={},saveViewState:d=!0,keepCurrentModel:h=!1,width:m="100%",height:f="100%",className:x,wrapperProps:S={},beforeMount:g=ji,onMount:y=ji,onChange:_,onValidate:w=ji}){let[z,k]=C.useState(!1),[E,N]=C.useState(!0),q=C.useRef(null),T=C.useRef(null),D=C.useRef(null),W=C.useRef(y),$=C.useRef(g),Z=C.useRef(),F=C.useRef(r),X=IS(o),ie=C.useRef(!1),R=C.useRef(!1);Cg(()=>{let V=wg.init();return V.then(H=>(q.current=H)&&N(!1)).catch(H=>(H==null?void 0:H.type)!=="cancelation"&&console.error("Monaco initialization: error:",H)),()=>T.current?b():V.cancel()}),ct(()=>{var H,v,te,ce;let V=Pr(q.current,e||r||"",t||i||"",o||n||"");V!==((H=T.current)==null?void 0:H.getModel())&&(d&&Bo.set(X,(v=T.current)==null?void 0:v.saveViewState()),(te=T.current)==null||te.setModel(V),d&&((ce=T.current)==null||ce.restoreViewState(Bo.get(o))))},[o],z),ct(()=>{var V;(V=T.current)==null||V.updateOptions(u)},[u],z),ct(()=>{!T.current||r===void 0||(T.current.getOption(q.current.editor.EditorOption.readOnly)?T.current.setValue(r):r!==T.current.getValue()&&(R.current=!0,T.current.executeEdits("",[{range:T.current.getModel().getFullModelRange(),text:r,forceMoveMarkers:!0}]),T.current.pushUndoStop(),R.current=!1))},[r],z),ct(()=>{var H,v;let V=(H=T.current)==null?void 0:H.getModel();V&&i&&((v=q.current)==null||v.editor.setModelLanguage(V,i))},[i],z),ct(()=>{var V;s!==void 0&&((V=T.current)==null||V.revealLine(s))},[s],z),ct(()=>{var V;(V=q.current)==null||V.editor.setTheme(a)},[a],z);let U=C.useCallback(()=>{var V;if(!(!D.current||!q.current)&&!ie.current){$.current(q.current);let H=o||n,v=Pr(q.current,r||e||"",t||i||"",H||"");T.current=(V=q.current)==null?void 0:V.editor.create(D.current,{model:v,automaticLayout:!0,...u},c),d&&T.current.restoreViewState(Bo.get(H)),q.current.editor.setTheme(a),s!==void 0&&T.current.revealLine(s),k(!0),ie.current=!0}},[e,t,n,r,i,o,u,c,d,a,s]);C.useEffect(()=>{z&&W.current(T.current,q.current)},[z]),C.useEffect(()=>{!E&&!z&&U()},[E,z,U]),F.current=r,C.useEffect(()=>{var V,H;z&&_&&((V=Z.current)==null||V.dispose(),Z.current=(H=T.current)==null?void 0:H.onDidChangeModelContent(v=>{R.current||_(T.current.getValue(),v)}))},[z,_]),C.useEffect(()=>{if(z){let V=q.current.editor.onDidChangeMarkers(H=>{var te;let v=(te=T.current.getModel())==null?void 0:te.uri;if(v&&H.find(ce=>ce.path===v.path)){let ce=q.current.editor.getModelMarkers({resource:v});w==null||w(ce)}});return()=>{V==null||V.dispose()}}return()=>{}},[z,w]);function b(){var V,H;(V=Z.current)==null||V.dispose(),h?d&&Bo.set(o,T.current.saveViewState()):(H=T.current.getModel())==null||H.dispose(),T.current.dispose()}return Re.createElement(Tg,{width:m,height:f,isEditorReady:z,loading:l,_ref:D,className:x,wrapperProps:S})}var MS=DS,qS=C.memo(MS),OS=qS;function aa({value:e,onChange:t,height:n="400px",onRun:r,darkMode:i}){const o=C.useRef(null),a=C.useRef(r);C.useEffect(()=>{a.current=r},[r]);const s=C.useCallback(c=>{t(c||"")},[t]),l=C.useCallback((c,d)=>{o.current=c,c.addAction({id:"run-tests",label:"Run Tests",keybindings:[d.KeyMod.Shift|d.KeyCode.Enter],run:()=>{a.current&&a.current()}})},[]),u=i?"vs-dark":"light";return p.jsx("div",{className:"h-full border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden shadow-sm",children:p.jsx(OS,{height:n,defaultLanguage:"python",theme:u,value:e,onChange:s,onMount:l,options:{minimap:{enabled:!1},fontSize:14,lineNumbers:"on",scrollBeyondLastLine:!1,automaticLayout:!0,tabSize:4,wordWrap:"on",padding:{top:16,bottom:16},scrollbar:{verticalScrollbarSize:8,horizontalScrollbarSize:8}}})})}function WS(e){const t=e.filter(n=>!n.hidden);return t.length===0?"":t.map((n,r)=>`# Test ${r+1}: ${n.description}
input: ${n.input}
expected: ${n.expected}`).join(`

`)}function VS(e,t){const n=t.filter(o=>o.hidden);return e.trim()?[...e.split(/\n\n+/).filter(o=>o.trim()).map((o,a)=>{const s=o.split(`
`);let l=`Test ${a+1}`,u="",c="";for(const d of s)if(d.startsWith("# Test")){const h=d.match(/# Test \d+: (.+)/);h&&(l=h[1])}else d.startsWith("input:")?u=d.replace("input:","").trim():d.startsWith("expected:")&&(c=d.replace("expected:","").trim());return{id:String(a+1),description:l,input:u,expected:c,hidden:!1}}).filter(o=>o.input&&o.expected),...n]:n}function sa({output:e,isLoading:t=!1,testCases:n,onTestCasesChange:r,problemId:i,resetKey:o=0,testResults:a}){const s=C.useRef(null),[l,u]=C.useState("tests"),[c,d]=C.useState(""),h=C.useRef(void 0),m=C.useRef(0),f=(n==null?void 0:n.filter(z=>!z.hidden).length)??0,x=(n==null?void 0:n.filter(z=>z.hidden).length)??0;C.useEffect(()=>{if(n&&n.length>0){const z=i!==h.current,k=o!==m.current,E=n.filter(q=>!q.hidden),N=c===""&&E.length>0;(z||k||N)&&(d(WS(n)),h.current=i,m.current=o,z&&u("tests"))}},[n,i,o,c]),C.useEffect(()=>{a&&a.length>0&&u("results")},[a]),C.useEffect(()=>{e.length>0&&(!a||a.length===0)&&u("output")},[e,a]),C.useEffect(()=>{var z;l==="output"&&((z=s.current)==null||z.scrollIntoView({behavior:"smooth"}))},[e,l]);const S=z=>{if(d(z),r&&n){const k=VS(z,n);r(k)}},g=n&&f>0,y=(a==null?void 0:a.filter(z=>z.passed).length)??0,_=(a==null?void 0:a.length)??0,w=(a==null?void 0:a.filter(z=>!z.hidden))??[];return p.jsxs("div",{className:"bg-white dark:bg-dark-800 border border-gray-200 dark:border-dark-500 rounded-lg h-full flex flex-col",children:[p.jsxs("div",{className:"flex items-center justify-between px-2 py-1 border-b border-gray-200 dark:border-dark-500 bg-gray-50 dark:bg-dark-900",children:[p.jsxs("div",{className:"flex",children:[g&&p.jsxs("button",{onClick:()=>u("tests"),className:`px-3 py-1.5 text-sm font-medium rounded-t transition-colors ${l==="tests"?"text-primary-600 dark:text-primary-300 bg-white dark:bg-dark-800 border-t border-l border-r border-gray-200 dark:border-dark-500 -mb-px":"text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100"}`,children:["Tests (",f,")"]}),p.jsx("button",{onClick:()=>u("output"),className:`px-3 py-1.5 text-sm font-medium rounded-t transition-colors ${l==="output"?"text-primary-600 dark:text-primary-300 bg-white dark:bg-dark-800 border-t border-l border-r border-gray-200 dark:border-dark-500 -mb-px":"text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100"}`,children:"Output"}),a&&a.length>0&&p.jsxs("button",{onClick:()=>u("results"),className:`px-3 py-1.5 text-sm font-medium rounded-t transition-colors ${l==="results"?"text-primary-600 dark:text-primary-300 bg-white dark:bg-dark-800 border-t border-l border-r border-gray-200 dark:border-dark-500 -mb-px":"text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100"}`,children:["Results (",y,"/",_,")"]})]}),t&&p.jsxs("div",{className:"flex items-center gap-2",children:[p.jsx("div",{className:"w-2 h-2 bg-primary-500 rounded-full animate-pulse"}),p.jsx("span",{className:"text-xs text-gray-500 dark:text-dark-200",children:"Running..."})]})]}),l==="tests"&&g?p.jsxs("div",{className:"flex-1 flex flex-col overflow-hidden",children:[p.jsx("textarea",{value:c,onChange:z=>S(z.target.value),className:"flex-1 p-4 font-mono text-sm bg-dark-900 text-dark-200 resize-none focus:outline-none",placeholder:`# Test 1: Description
input: your_function(args)
expected: expected_result`,spellCheck:!1}),x>0&&p.jsxs("div",{className:"px-4 py-2 bg-dark-800 text-dark-300 text-xs border-t border-dark-500",children:["+ ",x," hidden test",x>1?"s":""," (not editable)"]})]}):l==="results"&&a&&a.length>0?p.jsxs("div",{className:"flex-1 overflow-auto",children:[p.jsxs("div",{className:"px-4 py-3 border-b border-gray-200 dark:border-dark-500 flex items-center justify-between bg-gray-50 dark:bg-dark-900",children:[p.jsx("span",{className:"font-medium text-gray-900 dark:text-dark-100 text-sm",children:"Test Results"}),p.jsxs("span",{className:`text-sm font-medium font-mono ${y===_?"text-green-600 dark:text-green-400":"text-yellow-600 dark:text-yellow-400"}`,children:[y,"/",_," passed"]})]}),p.jsxs("div",{className:"divide-y divide-gray-200 dark:divide-dark-500",children:[w.map(z=>p.jsxs("div",{className:"p-4",children:[p.jsxs("div",{className:"flex items-center gap-2 mb-2",children:[p.jsx("span",{className:`w-5 h-5 rounded-full flex items-center justify-center text-xs ${z.passed?"bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400":"bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400"}`,children:z.passed?"✓":"✗"}),p.jsxs("span",{className:"text-gray-700 dark:text-dark-100 text-sm font-medium",children:["Test ",z.id,": ",z.description]})]}),!z.passed&&p.jsxs("div",{className:"ml-7 space-y-1 text-sm",children:[p.jsxs("div",{children:[p.jsx("span",{className:"text-gray-500 dark:text-dark-300",children:"Expected: "}),p.jsx("span",{className:"text-green-600 dark:text-green-400 font-mono",children:z.expected})]}),p.jsxs("div",{children:[p.jsx("span",{className:"text-gray-500 dark:text-dark-300",children:"Actual: "}),p.jsx("span",{className:"text-red-600 dark:text-red-400 font-mono",children:z.actual})]})]})]},z.id)),a.some(z=>z.hidden)&&p.jsxs("div",{className:"p-4 text-gray-500 dark:text-dark-300 text-sm italic",children:["+ ",a.filter(z=>z.hidden).length," hidden test(s)",a.filter(z=>z.hidden&&z.passed).length===a.filter(z=>z.hidden).length?p.jsx("span",{className:"text-green-600 dark:text-green-400 ml-2",children:"(all passed)"}):p.jsxs("span",{className:"text-red-600 dark:text-red-400 ml-2",children:["(",a.filter(z=>z.hidden&&!z.passed).length," failed)"]})]})]})]}):p.jsxs("div",{className:"flex-1 overflow-auto p-4 font-mono text-sm bg-dark-900 rounded-b-lg",children:[e.length===0?p.jsx("div",{className:"text-dark-300 italic",children:"Output will appear here when you run your code..."}):e.map((z,k)=>p.jsx("div",{className:`py-0.5 ${z.startsWith("[Error]")?"text-red-400":z.includes("PASSED")?"text-green-400":z.includes("FAILED")?"text-red-400":"text-dark-200"}`,children:z.startsWith("Test")?p.jsxs("span",{children:[z.includes("PASSED")?"✓ ":z.includes("FAILED")?"✗ ":"",z]}):p.jsxs("span",{children:["> ",z]})},k)),p.jsx("div",{ref:s})]})]})}function vh({problem:e}){return p.jsxs("div",{className:"prose prose-sm max-w-none overflow-hidden",children:[p.jsxs("div",{className:"flex items-center gap-3 mb-4",children:[p.jsx("h1",{className:"text-xl font-bold text-gray-900 dark:text-dark-100 m-0",children:e.title}),p.jsx("span",{className:`px-2 py-0.5 rounded text-xs font-medium whitespace-nowrap ${e.difficulty==="easy"?"bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400":e.difficulty==="medium"?"bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400":"bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"}`,children:e.difficulty})]}),p.jsx(B0,{remarkPlugins:[hg],components:{h2:({children:t})=>p.jsx("h2",{className:"text-lg font-semibold text-gray-900 dark:text-dark-100 mt-6 mb-3",children:t}),h3:({children:t})=>p.jsx("h3",{className:"text-base font-medium text-gray-700 dark:text-dark-200 mt-4 mb-2",children:t}),p:({children:t})=>p.jsx("p",{className:"text-gray-600 dark:text-dark-200 mb-3 leading-relaxed",children:t}),code:({className:t,children:n})=>(t==null?void 0:t.includes("language-"))?p.jsx("pre",{className:"bg-gray-100 dark:bg-dark-900 rounded-lg p-4 overflow-x-auto my-3",children:p.jsx("code",{className:"text-sm text-gray-800 dark:text-dark-100",children:n})}):p.jsx("code",{className:"bg-gray-100 dark:bg-dark-600 px-1.5 py-0.5 rounded text-primary-600 dark:text-primary-300 text-sm break-words",children:n}),ul:({children:t})=>p.jsx("ul",{className:"list-disc list-inside text-gray-600 dark:text-dark-200 space-y-1 mb-3",children:t}),li:({children:t})=>p.jsx("li",{className:"text-gray-600 dark:text-dark-200",children:t}),table:({children:t})=>p.jsx("div",{className:"overflow-x-auto my-4",children:p.jsx("table",{className:"min-w-full border-collapse border border-gray-300 dark:border-dark-500 text-sm",children:t})}),thead:({children:t})=>p.jsx("thead",{className:"bg-gray-100 dark:bg-dark-600",children:t}),tbody:({children:t})=>p.jsx("tbody",{className:"divide-y divide-gray-200 dark:divide-dark-500",children:t}),tr:({children:t})=>p.jsx("tr",{children:t}),th:({children:t})=>p.jsx("th",{className:"border border-gray-300 dark:border-dark-500 px-3 py-2 text-left font-semibold text-gray-700 dark:text-dark-200",children:t}),td:({children:t})=>p.jsx("td",{className:"border border-gray-300 dark:border-dark-500 px-3 py-2 text-gray-600 dark:text-dark-200",children:t})},children:e.description})]})}function kh({examples:e}){return p.jsxs("div",{children:[p.jsx("h3",{className:"text-base font-medium text-gray-900 dark:text-dark-100 mb-3",children:"Examples"}),p.jsx("div",{className:"space-y-4",children:e.map((t,n)=>p.jsx("div",{className:"bg-white dark:bg-dark-800 rounded-lg p-4 border border-gray-200 dark:border-dark-500 shadow-sm",children:p.jsxs("div",{className:"space-y-2",children:[p.jsxs("div",{className:"overflow-hidden",children:[p.jsx("span",{className:"text-gray-500 dark:text-dark-200 text-sm",children:"Input:"}),p.jsx("pre",{className:"mt-1 text-sm text-primary-600 dark:text-primary-300 font-mono whitespace-pre-wrap break-all",children:t.input})]}),p.jsxs("div",{className:"overflow-hidden",children:[p.jsx("span",{className:"text-gray-500 dark:text-dark-200 text-sm",children:"Output:"}),p.jsx("pre",{className:"mt-1 text-sm text-green-600 dark:text-green-400 font-mono whitespace-pre-wrap break-all",children:t.output})]}),t.explanation&&p.jsxs("div",{className:"pt-2 border-t border-gray-200 dark:border-dark-500",children:[p.jsx("span",{className:"text-gray-500 dark:text-dark-200 text-sm",children:"Explanation: "}),p.jsx("span",{className:"text-gray-600 dark:text-dark-200 text-sm",children:t.explanation})]})]})},n))})]})}function wh({hints:e,solution:t,onLoadToEditor:n,onShowSolution:r}){const[i,o]=C.useState(0),[a,s]=C.useState(!1),[l,u]=C.useState(!1),c=()=>{i<e.length&&o(f=>f+1)},d=()=>{s(!0),r==null||r()},h=async()=>{try{await navigator.clipboard.writeText(t),u(!0),setTimeout(()=>u(!1),2e3)}catch{const f=document.createElement("textarea");f.value=t,document.body.appendChild(f),f.select(),document.execCommand("copy"),document.body.removeChild(f),u(!0),setTimeout(()=>u(!1),2e3)}},m=()=>{n&&n(t)};return p.jsxs("div",{className:"space-y-4",children:[p.jsxs("div",{children:[p.jsxs("div",{className:"flex items-center justify-between mb-3",children:[p.jsx("h3",{className:"text-base font-medium text-gray-900 dark:text-dark-100",children:"Hints"}),i<e.length&&p.jsxs("button",{onClick:c,className:"text-sm text-primary-500 hover:text-primary-400 transition-colors",children:["Show Hint (",i,"/",e.length,")"]})]}),i>0?p.jsx("div",{className:"space-y-2",children:e.slice(0,i).map((f,x)=>p.jsx("div",{className:"bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800/50 rounded-lg p-3",children:p.jsxs("div",{className:"flex items-start gap-2",children:[p.jsx("span",{className:"text-yellow-600 flex-shrink-0",children:"💡"}),p.jsx("span",{className:"text-gray-700 dark:text-dark-200 text-sm break-words overflow-hidden",children:f})]})},x))}):p.jsx("div",{className:"text-gray-400 dark:text-dark-300 text-sm italic",children:'Click "Show Hint" if you need help'})]}),p.jsxs("div",{children:[p.jsxs("div",{className:"flex items-center justify-between mb-3",children:[p.jsx("h3",{className:"text-base font-medium text-gray-900 dark:text-dark-100",children:"Solution"}),!a&&i>=e.length&&p.jsx("button",{onClick:d,className:"text-sm text-primary-500 hover:text-primary-400 transition-colors",children:"Reveal Solution"})]}),a?p.jsxs("div",{className:"bg-white dark:bg-dark-800 rounded-lg overflow-hidden border border-gray-200 dark:border-dark-500",children:[p.jsxs("div",{className:"px-4 py-2 bg-gray-100 dark:bg-dark-900 border-b border-gray-200 dark:border-dark-500 flex items-center justify-between",children:[p.jsx("span",{className:"text-sm text-gray-600 dark:text-dark-200",children:"Solution Code"}),p.jsxs("div",{className:"flex items-center gap-2",children:[p.jsx("button",{onClick:h,className:"px-3 py-1 text-xs font-medium rounded-md bg-gray-200 dark:bg-dark-600 text-gray-700 dark:text-dark-200 hover:bg-gray-300 dark:hover:bg-dark-500 transition-colors",children:l?"Copied!":"Copy to Clipboard"}),n&&p.jsx("button",{onClick:m,className:"px-3 py-1 text-xs font-medium rounded-md bg-primary-500 text-white hover:bg-primary-600 transition-colors",children:"Load into Editor"})]})]}),p.jsx("pre",{className:"p-4 text-sm text-dark-200 font-mono overflow-x-auto bg-dark-900",children:t})]}):i<e.length?p.jsx("div",{className:"text-gray-400 dark:text-dark-300 text-sm italic",children:"Reveal all hints first to unlock the solution"}):p.jsx("div",{className:"text-gray-400 dark:text-dark-300 text-sm italic",children:'Click "Reveal Solution" to see the answer'})]})]})}function BS({show:e,onDismiss:t}){return C.useEffect(()=>{if(e){const n=setTimeout(t,3e3);return()=>clearTimeout(n)}},[e,t]),e?p.jsx("div",{className:"fixed top-16 left-0 right-0 z-40 animate-slide-down",children:p.jsx("div",{className:"bg-green-500 text-white px-4 py-3 shadow-lg",children:p.jsxs("div",{className:"max-w-7xl mx-auto flex items-center justify-between",children:[p.jsxs("div",{className:"flex items-center gap-2",children:[p.jsx("span",{className:"text-lg",children:"✅"}),p.jsx("span",{className:"font-medium",children:"All tests passed! Great work!"})]}),p.jsx("button",{onClick:t,className:"text-white/80 hover:text-white transition-colors text-sm",children:"Dismiss"})]})})}):null}const Th="ml-lab-language",US={numpy:"NumPy",pytorch:"PyTorch"};function HS(){var tt;const{sectionId:e,problemId:t}=Xf(),{getProblemProgress:n,saveProblemCode:r,updateProblemStatus:i,markSolutionViewed:o}=lo(),{isLoading:a,isReady:s,output:l,runTests:u,stop:c,clearOutput:d}=yg(),h=t?xb(t):void 0,m=Cn.find(Y=>Y.id===e),[f,x]=C.useState(()=>localStorage.getItem(Th)==="pytorch"?"pytorch":"numpy"),[S,g]=C.useState(null),_=(S&&S.problemId===(h==null?void 0:h.id)?S.language:(m==null?void 0:m.defaultLanguage)??f)==="pytorch"&&(h!=null&&h.pytorch)?"pytorch":"numpy",w=_==="pytorch"&&(h!=null&&h.pytorch)?h.pytorch:h,z=_==="pytorch"&&((tt=h==null?void 0:h.pytorch)==null?void 0:tt.hints)||(h==null?void 0:h.hints)||[],[k,E]=C.useState(""),[N,q]=C.useState([]),[T,D]=C.useState(!1),[W,$]=C.useState(!1),[Z,F]=C.useState([]),[X,ie]=C.useState(0),[R,U]=C.useState(()=>!localStorage.getItem("ml-lab-shortcut-hint-dismissed")),[b,V]=C.useState(!1),[H,v]=C.useState(null),te=C.useRef(!1),[ce,se]=C.useState("description"),[et,O]=C.useState(()=>window.innerWidth<1024),[A,I]=C.useState("problem"),{isDark:Q}=Td(),xe=C.useRef(null),fe=C.useRef(null);C.useEffect(()=>{if(h&&w&&e){const Y=`${e}/${h.id}/${_}`;if(xe.current===Y)return;xe.current=Y;const Ce=n(e,h.id),mn=_==="numpy"?Ce.code:Ce.codes[_];E(mn||w.starterCode),F(w.testCases),q([]),v(null),ie(Vt=>Vt+1),fe.current!==h.id&&(fe.current=h.id,se("description"))}},[h,w,_,e,n]),C.useEffect(()=>{if(h&&e){const Y=n(e,h.id);te.current=Y.status==="completed"}},[h==null?void 0:h.id,e,n]);const Ae=C.useRef(null);C.useEffect(()=>{Ae.current=xe.current},[k]),C.useEffect(()=>{if(k&&h&&w&&e&&k!==w.starterCode&&Ae.current===`${e}/${h.id}/${_}`){const Y=setTimeout(()=>{r(e,h.id,k,_)},1e3);return()=>clearTimeout(Y)}},[k,h,w,_,e,r]);const yt=C.useCallback(Y=>{Y===_||!h||!w||!e||(k&&k!==w.starterCode&&r(e,h.id,k,_),localStorage.setItem(Th,Y),x(Y),g({problemId:h.id,language:Y}))},[_,h,w,e,k,r]);C.useEffect(()=>{if(R){const Y=setTimeout(()=>{U(!1),localStorage.setItem("ml-lab-shortcut-hint-dismissed","true")},1e4);return()=>clearTimeout(Y)}},[R]);const $t=C.useCallback(()=>{U(!1),localStorage.setItem("ml-lab-shortcut-hint-dismissed","true")},[]);C.useEffect(()=>{const Y=()=>O(window.innerWidth<1024);return window.addEventListener("resize",Y),()=>window.removeEventListener("resize",Y)},[]);const Et=C.useCallback(Y=>{E(Y)},[]),mo=C.useCallback(()=>{w&&(!(k!==w.starterCode)||window.confirm("Reset your code? Your changes will be lost."))&&(E(w.starterCode),q([]),v(null),F(w.testCases),ie(Ce=>Ce+1),d())},[w,k,d]),fo=Y=>{const Ce=[...Y.matchAll(/def\s+(\w+)\s*\(/g)];return Ce.length>0?Ce[Ce.length-1][1]:"solution"},Vn=C.useCallback(async()=>{if(!(!h||!w||!s||!e||T)){D(!0),q([]),v(null);try{const Y=fo(w.starterCode),Ce=await u(k,Z,Y,_);if(q(Ce),!Ce.every(Vt=>Vt.passed))return;if(JSON.stringify(Z)!==JSON.stringify(w.testCases)&&!(await u(k,w.testCases,Y,_,{silent:!0})).every(Xe=>Xe.passed)){v("Your edited tests pass, but the original tests don't yet. Reset the tests to see which ones fail.");return}i(e,h.id,"completed",_),te.current||(V(!0),te.current=!0)}finally{D(!1)}}},[h,w,s,T,e,k,Z,_,u,i]),go=C.useCallback(()=>{h&&e&&o(e,h.id)},[h,e,o]);if(h&&h.section!==e)return p.jsx(Gf,{replace:!0,to:`/problem/${h.section}/${h.id}`});if(!h||!m||!w)return p.jsxs("div",{className:"text-center py-12",children:[p.jsx("h1",{className:"text-2xl font-bold text-gray-900 dark:text-dark-100 mb-4",children:"Problem Not Found"}),p.jsx(Ye,{to:"/",className:"text-primary-500 hover:text-primary-400",children:"Return to Home"})]});const cr=m.problems.indexOf(h.id),_o=cr>0?m.problems[cr-1]:void 0,hn=m.problems[cr+1],hs=h.description.replace(/[#*`]/g,"").split(`
`).filter(Y=>Y.trim()).slice(0,2).join(" ").substring(0,155),P=R?p.jsxs("div",{className:"absolute top-full left-1/2 -translate-x-1/2 mt-2 px-3 py-2 bg-dark-800 dark:bg-dark-600 text-white text-xs rounded-lg shadow-lg whitespace-nowrap z-50",children:[p.jsx("div",{className:"absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-dark-800 dark:bg-dark-600 rotate-45"}),p.jsx("span",{children:"Pro tip: "}),p.jsx("span",{className:"font-medium",children:"Shift+Enter"}),p.jsx("span",{children:" runs tests"}),p.jsx("button",{onClick:$t,className:"ml-2 text-gray-400 hover:text-white underline",children:"Got it"})]}):null,M=T?p.jsxs("button",{onClick:c,className:"px-4 py-1.5 bg-red-500/90 text-white text-sm font-medium rounded-lg hover:bg-red-600 transition-colors flex items-center gap-2",title:"Stop running code",children:[p.jsx("span",{className:"w-2.5 h-2.5 bg-white rounded-sm","aria-hidden":"true"}),"Stop"]}):p.jsx("button",{onClick:Vn,disabled:!s,className:"px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-lg hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:s?"Run Tests":"Loading Python…"}),G=h.pytorch?p.jsx("div",{role:"radiogroup","aria-label":"Solution language",className:"flex items-center p-0.5 bg-gray-200 dark:bg-dark-700 border border-gray-300 dark:border-dark-500 rounded-lg",children:["numpy","pytorch"].map(Y=>p.jsx("button",{role:"radio","aria-checked":_===Y,onClick:()=>yt(Y),disabled:T,className:`px-2.5 py-1 text-xs font-medium rounded-md transition-colors disabled:cursor-not-allowed ${_===Y?"bg-white dark:bg-dark-500 text-gray-900 dark:text-dark-100 shadow-sm":"text-gray-500 dark:text-dark-200 hover:text-gray-800 dark:hover:text-dark-100"}`,children:US[Y]},Y))}):null,ee=H||_==="pytorch"&&h.pytorch?p.jsxs("div",{className:"px-4 py-1.5 space-y-1 text-xs border-b border-gray-200 dark:border-dark-500 bg-gray-50 dark:bg-dark-900",children:[_==="pytorch"&&h.pytorch&&p.jsxs("p",{className:"text-gray-500 dark:text-dark-200",children:[p.jsx("span",{className:"font-medium text-gray-700 dark:text-dark-100",children:"PyTorch track"})," ","runs on an in-browser, NumPy-backed subset of the torch API.",h.pytorch.note?` ${h.pytorch.note}`:""]}),H&&p.jsx("p",{className:"text-amber-600 dark:text-amber-400",children:H})]}):null,le=p.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-gray-50 dark:bg-dark-800 border-b border-gray-200 dark:border-dark-500",children:[p.jsxs("div",{className:"flex items-center gap-2",children:[G,M,p.jsx("span",{className:"text-xs text-gray-400 dark:text-dark-300",children:"or"}),p.jsxs("div",{className:"relative",children:[p.jsxs("div",{className:"flex items-center gap-1 text-gray-500 dark:text-dark-200",children:[p.jsx("kbd",{className:"px-1.5 py-0.5 text-xs font-mono bg-gray-100 dark:bg-dark-600 border border-gray-300 dark:border-dark-500 rounded shadow-sm",children:"Shift"}),p.jsx("span",{className:"text-xs",children:"+"}),p.jsx("kbd",{className:"px-1.5 py-0.5 text-xs font-mono bg-gray-100 dark:bg-dark-600 border border-gray-300 dark:border-dark-500 rounded shadow-sm",children:"Enter"})]}),P]})]}),p.jsxs("div",{className:"flex items-center gap-2",children:[N.length>0&&N.every(Y=>Y.passed)&&!H&&hn&&p.jsx(Ye,{to:`/problem/${e}/${hn}`,className:"px-4 py-1.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-sm font-medium rounded-lg hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors",children:"Next Problem →"}),p.jsx("button",{onClick:()=>$(!W),className:"px-4 py-1.5 bg-gray-200 dark:bg-dark-600 text-gray-700 dark:text-dark-200 text-sm font-medium rounded-lg hover:bg-gray-300 dark:hover:bg-dark-500 transition-colors",title:W?"Restore layout":"Maximize editor",children:W?"⊟ Restore":"⊞ Maximize"}),p.jsx("button",{onClick:mo,className:"px-4 py-1.5 bg-gray-200 dark:bg-dark-600 text-gray-700 dark:text-dark-200 text-sm font-medium rounded-lg hover:bg-gray-300 dark:hover:bg-dark-500 transition-colors",children:"Reset"})]})]});return p.jsxs("div",{className:"h-[calc(100dvh-4rem)] flex flex-col -m-6",children:[p.jsx(ei,{title:`${h.title} - ${m.title}`,description:`${hs}... Practice this ${h.difficulty} ML coding problem with instant feedback.`,canonical:`/problem/${e}/${t}`,keywords:`${h.title}, ${m.title}, machine learning, coding practice, ${h.difficulty}`}),p.jsx(BS,{show:b,onDismiss:()=>V(!1)}),p.jsxs("div",{className:"flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-4 sm:px-6 py-3 bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-500",children:[p.jsxs("div",{className:"flex items-center gap-3 min-w-0",children:[p.jsxs(Ye,{to:`/section/${e}`,className:"shrink-0 text-gray-500 dark:text-dark-200 hover:text-gray-900 dark:hover:text-dark-100 transition-colors",children:["← ",p.jsx("span",{className:"hidden sm:inline",children:m.title})]}),p.jsx("span",{className:"hidden sm:inline text-gray-300 dark:text-dark-500",children:"/"}),p.jsx("span",{className:"text-gray-900 dark:text-dark-100 font-medium truncate",children:h.title})]}),p.jsxs("div",{className:"flex items-center gap-3",children:[p.jsxs("div",{className:"flex items-center gap-1",children:[_o?p.jsx(Ye,{to:`/problem/${e}/${_o}`,className:"px-2 py-1 text-sm text-gray-600 dark:text-dark-200 hover:text-gray-900 dark:hover:text-dark-100 hover:bg-gray-100 dark:hover:bg-dark-600 rounded transition-colors",children:"← Prev"}):p.jsx("span",{className:"px-2 py-1 text-sm text-gray-300 dark:text-dark-400 cursor-default",children:"← Prev"}),p.jsxs("span",{className:"text-xs text-gray-400 dark:text-dark-300 font-mono",children:["(",cr+1,"/",m.problems.length,")"]}),hn?p.jsx(Ye,{to:`/problem/${e}/${hn}`,className:"px-2 py-1 text-sm text-gray-600 dark:text-dark-200 hover:text-gray-900 dark:hover:text-dark-100 hover:bg-gray-100 dark:hover:bg-dark-600 rounded transition-colors",children:"Next →"}):p.jsx("span",{className:"px-2 py-1 text-sm text-gray-300 dark:text-dark-400 cursor-default",children:"Next →"})]}),p.jsx("span",{className:"hidden sm:inline text-gray-300 dark:text-dark-500",children:"|"}),!s&&p.jsxs("span",{className:"text-gray-500 dark:text-dark-200 text-sm hidden sm:flex items-center gap-2",children:[p.jsx("div",{className:"w-2 h-2 bg-yellow-500 rounded-full animate-pulse"}),a?"Loading Python...":"Python ready"]}),s&&p.jsxs("span",{className:"text-gray-500 dark:text-dark-200 text-sm hidden sm:flex items-center gap-2",children:[p.jsx("div",{className:"w-2 h-2 bg-accent-500 rounded-full"}),"Python ready"]})]})]}),et?p.jsxs("div",{className:"flex flex-col flex-1 overflow-hidden bg-white dark:bg-dark-800",children:[p.jsxs("div",{className:"flex border-b border-gray-200 dark:border-dark-500 bg-gray-50 dark:bg-dark-900",children:[p.jsx("button",{onClick:()=>I("problem"),className:`flex-1 px-4 py-2.5 text-sm font-medium text-center transition-colors ${A==="problem"?"text-primary-600 dark:text-primary-300 border-b-2 border-primary-500 bg-white dark:bg-dark-800":"text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100"}`,children:"Problem"}),p.jsx("button",{onClick:()=>I("code"),className:`flex-1 px-4 py-2.5 text-sm font-medium text-center transition-colors ${A==="code"?"text-primary-600 dark:text-primary-300 border-b-2 border-primary-500 bg-white dark:bg-dark-800":"text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100"}`,children:"Code"})]}),A==="problem"?p.jsxs("div",{className:"flex-1 overflow-y-auto p-4",children:[p.jsx(vh,{problem:h}),p.jsx(kh,{examples:h.examples}),p.jsx(wh,{hints:z,solution:w.solution,onShowSolution:go,onLoadToEditor:Y=>{E(Y),I("code")}},`${h.id}-${_}`)]}):p.jsxs("div",{className:"flex-1 flex flex-col overflow-hidden",children:[p.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-gray-50 dark:bg-dark-900 border-b border-gray-200 dark:border-dark-500",children:[p.jsxs("div",{className:"flex items-center gap-2",children:[G,M,p.jsx("button",{onClick:mo,className:"px-3 py-1.5 bg-gray-200 dark:bg-dark-600 text-gray-700 dark:text-dark-200 text-sm font-medium rounded-lg hover:bg-gray-300 dark:hover:bg-dark-500 transition-colors",children:"Reset"})]}),N.length>0&&N.every(Y=>Y.passed)&&!H&&hn&&p.jsx(Ye,{to:`/problem/${e}/${hn}`,className:"px-3 py-1.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-sm font-medium rounded-lg hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors",children:"Next →"})]}),ee,p.jsx("div",{className:"flex-[3] overflow-hidden p-2",children:p.jsx(aa,{value:k,onChange:Et,height:"100%",onRun:Vn,darkMode:Q})}),p.jsx("div",{className:"flex-[2] overflow-auto p-2 space-y-2 bg-gray-50 dark:bg-dark-900 border-t border-gray-200 dark:border-dark-500",children:p.jsx(sa,{output:l,isLoading:T,testCases:Z,onTestCasesChange:F,problemId:t,resetKey:X,testResults:N})})]})]}):W?p.jsxs("div",{className:"flex flex-col flex-1 overflow-hidden bg-white dark:bg-dark-800",children:[le,ee,p.jsxs(qr,{className:"flex flex-1 overflow-hidden",sizes:[60,40],minSize:200,gutterSize:8,gutterAlign:"center",children:[p.jsx("div",{className:"h-full overflow-hidden p-4",children:p.jsx(aa,{value:k,onChange:Et,height:"100%",onRun:Vn,darkMode:Q})}),p.jsx("div",{className:"overflow-auto p-4 bg-gray-50 dark:bg-dark-900",children:p.jsx(sa,{output:l,isLoading:T,testCases:Z,onTestCasesChange:F,problemId:t,resetKey:X,testResults:N})})]})]}):p.jsxs(qr,{className:"flex flex-1 overflow-hidden",sizes:[40,60],minSize:300,gutterSize:8,gutterAlign:"center",children:[p.jsxs("div",{className:"problem-panel flex flex-col overflow-hidden bg-gray-50 dark:bg-dark-900 min-w-0",children:[p.jsxs("div",{className:"flex overflow-x-auto whitespace-nowrap border-b border-gray-200 dark:border-dark-500 bg-white dark:bg-dark-800 px-4 shrink-0",children:[p.jsx("button",{onClick:()=>se("description"),className:`whitespace-nowrap shrink-0 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${ce==="description"?"border-primary-500 text-primary-600 dark:text-primary-300":"border-transparent text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100 hover:border-gray-300 dark:hover:border-dark-400"}`,children:"Description"}),p.jsx("button",{onClick:()=>se("examples"),className:`whitespace-nowrap shrink-0 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${ce==="examples"?"border-primary-500 text-primary-600 dark:text-primary-300":"border-transparent text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100 hover:border-gray-300 dark:hover:border-dark-400"}`,children:"Examples"}),p.jsx("button",{onClick:()=>se("hints"),className:`whitespace-nowrap shrink-0 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${ce==="hints"?"border-primary-500 text-primary-600 dark:text-primary-300":"border-transparent text-gray-500 dark:text-dark-200 hover:text-gray-700 dark:hover:text-dark-100 hover:border-gray-300 dark:hover:border-dark-400"}`,children:"Hints & Solution"})]}),p.jsxs("div",{className:"flex-1 overflow-y-auto overflow-x-hidden p-6",children:[ce==="description"&&p.jsx(vh,{problem:h}),ce==="examples"&&p.jsx("div",{children:p.jsx(kh,{examples:h.examples})}),ce==="hints"&&p.jsx("div",{children:p.jsx(wh,{hints:z,solution:w.solution,onShowSolution:go,onLoadToEditor:E},`${h.id}-${_}`)})]})]}),p.jsxs("div",{className:"flex flex-col overflow-hidden bg-white dark:bg-dark-800",children:[le,ee,p.jsxs(qr,{className:"flex-1 overflow-hidden",direction:"vertical",sizes:[60,40],minSize:100,gutterSize:8,children:[p.jsx("div",{className:"h-full overflow-hidden p-4",children:p.jsx(aa,{value:k,onChange:Et,height:"100%",onRun:Vn,darkMode:Q})}),p.jsx("div",{className:"overflow-auto p-4 bg-gray-50 dark:bg-dark-900",children:p.jsx(sa,{output:l,isLoading:T,testCases:Z,onTestCasesChange:F,problemId:t,resetKey:X,testResults:N})})]})]})]})]})}const Ch="ml-interview-scratchpad",Sh=`import numpy as np

# Try out your code here!
# NumPy is already loaded and ready to use.

arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Sum: {np.sum(arr)}")
print(f"Mean: {np.mean(arr)}")
`;function QS(){const{isLoading:e,isReady:t,output:n,runCode:r,stop:i,clearOutput:o}=yg(),{isDark:a}=Td(),[s,l]=C.useState(""),[u,c]=C.useState(!1);C.useEffect(()=>{const f=localStorage.getItem(Ch);l(f||Sh)},[]),C.useEffect(()=>{if(s){const f=setTimeout(()=>{localStorage.setItem(Ch,s)},1e3);return()=>clearTimeout(f)}},[s]);const d=C.useCallback(f=>{l(f)},[]),h=C.useCallback(()=>{l(Sh),o()},[o]),m=C.useCallback(async()=>{if(t){c(!0),o();try{await r(s)}finally{c(!1)}}},[t,s,r,o]);return p.jsxs("div",{className:"h-[calc(100dvh-4rem)] flex flex-col -m-6",children:[p.jsx(ei,{title:"Python Scratchpad",description:"Free Python playground with NumPy pre-loaded. Run Python code directly in your browser - no setup required. Practice machine learning code instantly.",canonical:"/scratchpad",keywords:"Python playground, online Python, NumPy, code sandbox, ML practice, browser Python"}),p.jsxs("div",{className:"flex items-center justify-between px-6 py-3 bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-500",children:[p.jsxs("div",{className:"flex items-center gap-3",children:[p.jsx(Ye,{to:"/",className:"text-gray-500 dark:text-dark-200 hover:text-gray-900 dark:hover:text-dark-100 transition-colors",children:"Home"}),p.jsx("span",{className:"text-gray-300 dark:text-dark-500",children:"/"}),p.jsx("span",{className:"text-gray-900 dark:text-dark-100 font-medium",children:"Python Scratchpad"})]}),p.jsxs("div",{className:"flex items-center gap-3",children:[!t&&p.jsxs("span",{className:"text-gray-500 dark:text-dark-200 text-sm flex items-center gap-2",children:[p.jsx("div",{className:"w-2 h-2 bg-yellow-500 rounded-full animate-pulse"}),e?"Loading Python...":"Python ready"]}),t&&p.jsxs("span",{className:"text-gray-500 dark:text-dark-200 text-sm flex items-center gap-2",children:[p.jsx("div",{className:"w-2 h-2 bg-accent-500 rounded-full"}),"Python ready"]})]})]}),p.jsxs("div",{className:"flex flex-col flex-1 overflow-hidden bg-white dark:bg-dark-800",children:[p.jsxs("div",{className:"flex items-center justify-between px-4 py-2 bg-gray-50 dark:bg-dark-900 border-b border-gray-200 dark:border-dark-500",children:[p.jsxs("div",{className:"flex items-center gap-2",children:[u?p.jsxs("button",{onClick:i,className:"px-4 py-1.5 bg-red-500/90 text-white text-sm font-medium rounded-lg hover:bg-red-600 transition-colors flex items-center gap-2",title:"Stop running code",children:[p.jsx("span",{className:"w-2.5 h-2.5 bg-white rounded-sm","aria-hidden":"true"}),"Stop"]}):p.jsx("button",{onClick:m,disabled:!t,className:"px-4 py-1.5 bg-primary-500 text-white text-sm font-medium rounded-lg hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors",children:"Run Code"}),p.jsx("button",{onClick:h,className:"px-4 py-1.5 bg-gray-200 dark:bg-dark-600 text-gray-700 dark:text-dark-200 text-sm font-medium rounded-lg hover:bg-gray-300 dark:hover:bg-dark-500 transition-colors",children:"Reset"}),p.jsx("button",{onClick:o,className:"px-4 py-1.5 bg-gray-200 dark:bg-dark-600 text-gray-700 dark:text-dark-200 text-sm font-medium rounded-lg hover:bg-gray-300 dark:hover:bg-dark-500 transition-colors",children:"Clear Output"})]}),p.jsx("span",{className:"text-gray-400 dark:text-dark-300 text-sm",children:"Code is auto-saved"})]}),p.jsxs(qr,{className:"flex-1 overflow-hidden",direction:"vertical",sizes:[60,40],minSize:100,gutterSize:8,children:[p.jsx("div",{className:"h-full overflow-hidden p-4",children:p.jsx(aa,{value:s,onChange:d,height:"100%",darkMode:a})}),p.jsx("div",{className:"overflow-auto p-4 bg-gray-50 dark:bg-dark-900",children:p.jsx(sa,{output:n,isLoading:u})})]})]})]})}function XS(){return p.jsxs(p.Fragment,{children:[p.jsx(ei,{title:"Terms of Service",description:"Terms of Service for ML Coding Lab - an interactive platform for learning machine learning through hands-on coding exercises.",canonical:"/terms"}),p.jsxs("div",{className:"max-w-4xl mx-auto",children:[p.jsx("h1",{className:"text-3xl font-bold text-gray-900 mb-8",children:"Terms of Service"}),p.jsxs("div",{className:"prose prose-gray max-w-none",children:[p.jsxs("p",{className:"text-gray-600 mb-6",children:[p.jsx("strong",{children:"Last updated:"})," February 1, 2026"]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"1. Acceptance of Terms"}),p.jsx("p",{className:"text-gray-600 mb-4",children:'By accessing and using ML Coding Lab ("the Service"), you accept and agree to be bound by these Terms of Service. If you do not agree to these terms, please do not use the Service.'})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"2. Description of Service"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"ML Coding Lab is an educational platform that provides interactive machine learning coding exercises, tutorials, and learning materials. The Service allows users to:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Read problem descriptions and theoretical introductions"}),p.jsx("li",{children:"Write and execute Python code in the browser"}),p.jsx("li",{children:"Run tests and receive feedback on solutions"}),p.jsx("li",{children:"Track learning progress across sections"}),p.jsx("li",{children:"Use a Python scratchpad for experimentation"})]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"3. Intellectual Property Rights"}),p.jsxs("p",{className:"text-gray-600 mb-4",children:[p.jsx("strong",{children:"Software Code:"})," The source code of ML Coding Lab is licensed under the MIT License. You may use, modify, and distribute the code in accordance with the terms of that license."]}),p.jsxs("p",{className:"text-gray-600 mb-4",children:[p.jsx("strong",{children:"Educational Content:"})," All educational content, including but not limited to:"]}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Problem descriptions and explanations"}),p.jsx("li",{children:"Solutions and starter code for exercises"}),p.jsx("li",{children:"Learning materials and tutorials"}),p.jsx("li",{children:"Test cases and examples"})]}),p.jsx("p",{className:"text-gray-600 mb-4",children:"is copyrighted material owned by Siddharth Choudhary. This content may not be reproduced, distributed, modified, or used to create derivative works without explicit written permission from the copyright holder."})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"4. Permitted Use"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"You may use the Service for:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Personal learning and educational purposes"}),p.jsx("li",{children:"Preparing for interviews or assessments"}),p.jsx("li",{children:"Building your machine learning skills"})]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"5. Prohibited Use"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"You may NOT:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Copy, reproduce, or redistribute the educational content"}),p.jsx("li",{children:"Use the content for commercial purposes without permission"}),p.jsx("li",{children:"Create competing services using our educational materials"}),p.jsx("li",{children:"Scrape or bulk download content from the Service"}),p.jsx("li",{children:"Remove or alter any copyright notices or attributions"}),p.jsx("li",{children:"Attempt to circumvent any security measures"})]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"6. User Code"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"Code that you write in the code editor is stored locally in your browser. We do not collect or store your code on our servers. You retain full ownership of any original code you create while using the Service."})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"7. Disclaimer of Warranties"}),p.jsx("p",{className:"text-gray-600 mb-4",children:'THE SERVICE IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED. WE DO NOT WARRANT THAT THE SERVICE WILL BE UNINTERRUPTED, ERROR-FREE, OR SECURE.'})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"8. Limitation of Liability"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"TO THE MAXIMUM EXTENT PERMITTED BY LAW, WE SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES ARISING FROM YOUR USE OF THE SERVICE."})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"9. Changes to Terms"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"We reserve the right to modify these Terms at any time. Changes will be effective immediately upon posting. Your continued use of the Service after changes constitutes acceptance of the modified Terms."})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"10. Contact"}),p.jsxs("p",{className:"text-gray-600 mb-4",children:["For questions about these Terms, please open an issue on our"," ",p.jsx("a",{href:"https://github.com/itzsid/ml-coding-lab",target:"_blank",rel:"noopener noreferrer",className:"text-primary-600 hover:text-primary-700",children:"GitHub repository"}),"."]})]})]})]})]})}function KS(){return p.jsxs(p.Fragment,{children:[p.jsx(ei,{title:"Privacy Policy",description:"Privacy Policy for ML Coding Lab - learn how we handle your data on our machine learning education platform.",canonical:"/privacy"}),p.jsxs("div",{className:"max-w-4xl mx-auto",children:[p.jsx("h1",{className:"text-3xl font-bold text-gray-900 mb-8",children:"Privacy Policy"}),p.jsxs("div",{className:"prose prose-gray max-w-none",children:[p.jsxs("p",{className:"text-gray-600 mb-6",children:[p.jsx("strong",{children:"Last updated:"})," February 1, 2026"]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Overview"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"ML Coding Lab is committed to protecting your privacy. This Privacy Policy explains how we collect, use, and protect information when you use our Service."}),p.jsxs("p",{className:"text-gray-600 mb-4",children:[p.jsx("strong",{children:"The short version:"})," We collect minimal data. Your code stays in your browser. We don't track you or sell your data."]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Information We Collect"}),p.jsx("h3",{className:"text-lg font-medium text-gray-700 mb-3",children:"Data Stored Locally (Your Browser)"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"The following data is stored only in your browser's localStorage and never sent to our servers:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsxs("li",{children:[p.jsx("strong",{children:"Progress data:"})," Which problems you've completed"]}),p.jsxs("li",{children:[p.jsx("strong",{children:"Saved code:"})," Your code solutions for each problem"]}),p.jsxs("li",{children:[p.jsx("strong",{children:"Scratchpad code:"})," Code you've written in the Python scratchpad"]})]}),p.jsx("p",{className:"text-gray-600 mb-4",children:"This data never leaves your device unless you explicitly export it."}),p.jsx("h3",{className:"text-lg font-medium text-gray-700 mb-3",children:"Feedback Submissions"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"When you submit feedback through our feedback form, we collect:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Your email address (optional, only if you provide it)"}),p.jsx("li",{children:"The feedback message you write"})]}),p.jsx("p",{className:"text-gray-600 mb-4",children:"This information is sent via Formspree and is used solely to respond to your feedback and improve the Service."}),p.jsx("h3",{className:"text-lg font-medium text-gray-700 mb-3",children:"Analytics and Hosting"}),p.jsxs("p",{className:"text-gray-600 mb-4",children:["ML Coding Lab is hosted on GitHub Pages. GitHub may collect standard server logs including IP addresses and page requests. Please refer to"," ",p.jsx("a",{href:"https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement",target:"_blank",rel:"noopener noreferrer",className:"text-primary-600 hover:text-primary-700",children:"GitHub's Privacy Statement"})," ","for more information."]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Information We Do NOT Collect"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"We do not require account registration"}),p.jsx("li",{children:"We do not use cookies for tracking"}),p.jsx("li",{children:"We do not use third-party analytics services"}),p.jsx("li",{children:"We do not collect personal information beyond what you voluntarily provide"}),p.jsx("li",{children:"We do not sell or share your data with third parties"})]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Code Execution"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"All Python code execution happens entirely in your browser using Pyodide (WebAssembly). Your code is never sent to any external server for execution. This means:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Your code remains private on your device"}),p.jsx("li",{children:"We cannot see or access the code you write"}),p.jsx("li",{children:"Execution results stay in your browser"})]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Data Retention"}),p.jsxs("p",{className:"text-gray-600 mb-4",children:[p.jsx("strong",{children:"Local data:"})," Data stored in your browser persists until you clear your browser's localStorage or use a browser cleaning tool."]}),p.jsxs("p",{className:"text-gray-600 mb-4",children:[p.jsx("strong",{children:"Feedback:"})," Feedback submissions are retained only as long as necessary to respond to your inquiry and improve the Service."]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Your Rights"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"You have the right to:"}),p.jsxs("ul",{className:"list-disc list-inside text-gray-600 mb-4 space-y-2",children:[p.jsx("li",{children:"Clear your local data at any time through your browser settings"}),p.jsx("li",{children:"Use the Service without providing any personal information"}),p.jsx("li",{children:"Request deletion of any feedback you've submitted"})]})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Children's Privacy"}),p.jsx("p",{className:"text-gray-600 mb-4",children:"The Service is intended for general audiences interested in learning machine learning. We do not knowingly collect personal information from children under 13. Since we don't require registration and store data locally, children can use the Service without providing any personal information."})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Changes to This Policy"}),p.jsx("p",{className:"text-gray-600 mb-4",children:'We may update this Privacy Policy from time to time. We will notify users of any material changes by updating the "Last updated" date at the top of this policy.'})]}),p.jsxs("section",{className:"mb-8",children:[p.jsx("h2",{className:"text-xl font-semibold text-gray-800 mb-4",children:"Contact"}),p.jsxs("p",{className:"text-gray-600 mb-4",children:["For questions about this Privacy Policy, please open an issue on our"," ",p.jsx("a",{href:"https://github.com/itzsid/ml-coding-lab",target:"_blank",rel:"noopener noreferrer",className:"text-primary-600 hover:text-primary-700",children:"GitHub repository"}),"."]})]})]})]})]})}function YS(){return p.jsx(zb,{children:p.jsxs(Q1,{children:[p.jsx(Qn,{path:"/",element:p.jsx(Ab,{})}),p.jsx(Qn,{path:"/section/:sectionId",element:p.jsx(v4,{})}),p.jsx(Qn,{path:"/problem/:sectionId/:problemId",element:p.jsx(HS,{})}),p.jsx(Qn,{path:"/scratchpad",element:p.jsx(QS,{})}),p.jsx(Qn,{path:"/terms",element:p.jsx(XS,{})}),p.jsx(Qn,{path:"/privacy",element:p.jsx(KS,{})})]})})}pl.createRoot(document.getElementById("root")).render(p.jsx(Re.StrictMode,{children:p.jsx(o0,{children:p.jsx(ex,{basename:"/ml-coding-lab",children:p.jsx(kb,{children:p.jsx(YS,{})})})})}));
